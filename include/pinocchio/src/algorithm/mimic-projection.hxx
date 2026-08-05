//
// Copyright (c) 2026 INRIA
//

#pragma once

// IWYU pragma: private, include "pinocchio/algorithm/mimic-projection.hpp"

#ifdef PINOCCHIO_LSP
  #undef PINOCCHIO_LSP
  #include "pinocchio/algorithm/mimic-projection.hpp"
#endif // PINOCCHIO_LSP

namespace pinocchio
{

  template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
  void buildMimicProjection(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_full,
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_reduced,
    MimicProjectionTpl<Scalar, Options> & projection)
  {
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef typename Model::JointModel JointModel;
    typedef JointModelMimicTpl<Scalar, Options, JointCollectionTpl> JointModelMimic;
    typedef MimicProjectionTpl<Scalar, Options> Projection;
    typedef typename Projection::JointCoupling JointCoupling;

    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      model_full.njoints, model_reduced.njoints,
      "The expanded and the reduced models do not have the same number of joints.");
    PINOCCHIO_CHECK_INPUT_ARGUMENT(
      model_full.mimicking_joints.empty(),
      "The expanded model must not contain any mimic joint.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      model_full.nv, model_reduced.nvExtended,
      "The expanded model dimension does not match the extended dimension of the reduced model.");

    projection.nq_full = model_full.nq;
    projection.nv_full = model_full.nv;
    projection.nq_reduced = model_reduced.nq;
    projection.nv_reduced = model_reduced.nv;

    projection.joint_couplings.clear();
    projection.joint_couplings.reserve(size_t(model_full.njoints - 1));

    projection.v_reduced_index.assign(size_t(model_full.nv), 0);
    projection.v_scaling.setOnes(model_full.nv);

    for (JointIndex j = 1; j < JointIndex(model_full.njoints); ++j)
    {
      const JointModel & jmodel_full = model_full.joints[j];
      const JointModel & jmodel_reduced = model_reduced.joints[j];

      JointCoupling coupling;
      coupling.joint_id = j;
      coupling.idx_q_full = jmodel_full.idx_q();
      coupling.nq_joint = jmodel_full.nq();
      coupling.idx_v_full = jmodel_full.idx_v();
      coupling.nv_joint = jmodel_full.nv();
      // For a mimicking joint, idx_q()/idx_v() point at the mimicked joint of the reduced model,
      // which is exactly the reduced coordinate the expanded one is an affine function of.
      coupling.idx_q_reduced = jmodel_reduced.idx_q();
      coupling.idx_v_reduced = jmodel_reduced.idx_v();
      coupling.scaling = Scalar(1);
      coupling.offset = Scalar(0);

      if (const JointModelMimic * jmodel_mimic = boost::get<JointModelMimic>(&jmodel_reduced))
      {
        coupling.scaling = jmodel_mimic->scaling();
        coupling.offset = jmodel_mimic->offset();
      }

      PINOCCHIO_CHECK_ARGUMENT_SIZE(
        coupling.nv_joint, jmodel_reduced.nvExtended(),
        "A joint of the expanded model does not match its counterpart in the reduced model.");

      for (int k = 0; k < coupling.nv_joint; ++k)
      {
        const int row = coupling.idx_v_full + k;
        projection.v_reduced_index[size_t(row)] = coupling.idx_v_reduced + k;
        projection.v_scaling[row] = coupling.scaling;
      }

      projection.joint_couplings.push_back(coupling);
    }

    projection.q_full.setZero(model_full.nq);
    projection.v_full.setZero(model_full.nv);
    projection.a_full.setZero(model_full.nv);
    projection.M_full.setZero(model_full.nv, model_full.nv);
    projection.contraction_tmp.setZero(model_reduced.nv, model_full.nv);
  }

  template<typename Scalar, int Options, typename MatrixType>
  void mimicProjectionMatrix(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<MatrixType> & G)
  {
    MatrixType & G_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixType, G);
    G_.setZero(projection.nv_full, projection.nv_reduced);

    for (Eigen::DenseIndex row = 0; row < projection.nv_full; ++row)
      G_(row, projection.v_reduced_index[size_t(row)]) = projection.v_scaling[row];
  }

  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorIn,
    typename ConfigVectorOut>
  void mimicExpandConfiguration(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_full,
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<ConfigVectorIn> & q_reduced,
    const Eigen::MatrixBase<ConfigVectorOut> & q_full)
  {
    typedef MimicProjectionTpl<Scalar, Options> Projection;
    typedef typename Projection::JointCoupling JointCoupling;

    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      q_reduced.size(), projection.nq_reduced, "The reduced configuration vector is not of the "
                                               "right size.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      q_full.size(), projection.nq_full, "The expanded configuration vector is not of the right "
                                         "size.");

    ConfigVectorOut & q_full_ = PINOCCHIO_EIGEN_CONST_CAST(ConfigVectorOut, q_full);

    for (typename Projection::JointCouplingVector::const_iterator it =
           projection.joint_couplings.begin();
         it != projection.joint_couplings.end(); ++it)
    {
      const JointCoupling & coupling = *it;
      // Same affine transform as the one JointModelMimicTpl applies internally, so that the
      // expanded configuration is the one the reduced forward kinematics implicitly uses.
      configVectorAffineTransform(
        model_full.joints[coupling.joint_id],
        q_reduced.segment(coupling.idx_q_reduced, coupling.nq_joint), coupling.scaling,
        coupling.offset, q_full_.segment(coupling.idx_q_full, coupling.nq_joint));
    }
  }

  template<typename Scalar, int Options, typename TangentVectorIn, typename TangentVectorOut>
  void mimicExpandTangent(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<TangentVectorIn> & x_reduced,
    const Eigen::MatrixBase<TangentVectorOut> & x_full)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      x_reduced.size(), projection.nv_reduced, "The reduced tangent vector is not of the right "
                                               "size.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      x_full.size(), projection.nv_full, "The expanded tangent vector is not of the right size.");

    TangentVectorOut & x_full_ = PINOCCHIO_EIGEN_CONST_CAST(TangentVectorOut, x_full);

    for (Eigen::DenseIndex row = 0; row < projection.nv_full; ++row)
      x_full_[row] = projection.v_scaling[row] * x_reduced[projection.v_reduced_index[size_t(row)]];
  }

  template<typename Scalar, int Options, typename VectorIn, typename VectorOut>
  void mimicContractCotangent(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<VectorIn> & f_full,
    const Eigen::MatrixBase<VectorOut> & f_reduced)
  {
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      f_full.size(), projection.nv_full, "The expanded cotangent vector is not of the right size.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      f_reduced.size(), projection.nv_reduced, "The reduced cotangent vector is not of the right "
                                               "size.");

    VectorOut & f_reduced_ = PINOCCHIO_EIGEN_CONST_CAST(VectorOut, f_reduced);
    f_reduced_.setZero();

    // Several expanded rows may scatter onto the same reduced one, hence the accumulation.
    for (Eigen::DenseIndex row = 0; row < projection.nv_full; ++row)
      f_reduced_[projection.v_reduced_index[size_t(row)]] +=
        projection.v_scaling[row] * f_full[row];
  }

  template<typename Scalar, int Options, typename MatrixIn, typename MatrixOut>
  void mimicContractMatrix(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<MatrixIn> & D_full,
    const Eigen::MatrixBase<MatrixOut> & D_reduced)
  {
    typedef MimicProjectionTpl<Scalar, Options> Projection;

    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      D_full.rows(), projection.nv_full, "The expanded matrix is not of the right size.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      D_full.cols(), projection.nv_full, "The expanded matrix is not of the right size.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      D_reduced.rows(), projection.nv_reduced, "The reduced matrix is not of the right size.");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      D_reduced.cols(), projection.nv_reduced, "The reduced matrix is not of the right size.");

    MatrixOut & D_reduced_ = PINOCCHIO_EIGEN_CONST_CAST(MatrixOut, D_reduced);
    typename Projection::MatrixXs & tmp = projection.contraction_tmp;

    // First pass: tmp = G^T D_e, gathering the expanded rows onto the reduced ones.
    tmp.setZero();
    for (Eigen::DenseIndex row = 0; row < projection.nv_full; ++row)
      tmp.row(projection.v_reduced_index[size_t(row)]) +=
        projection.v_scaling[row] * D_full.row(row);

    // Second pass: D_r = tmp G, gathering the expanded columns onto the reduced ones.
    D_reduced_.setZero();
    for (Eigen::DenseIndex col = 0; col < projection.nv_full; ++col)
      D_reduced_.col(projection.v_reduced_index[size_t(col)]) +=
        projection.v_scaling[col] * tmp.col(col);
  }

  namespace internal
  {
    /// \brief Symmetrizes the upper triangular joint space inertia matrix left by
    /// pinocchio::computeRNEADerivatives, then contracts the three expanded derivatives.
    template<
      typename Scalar,
      int Options,
      template<typename, int> class JointCollectionTpl,
      typename MatrixType1,
      typename MatrixType2,
      typename MatrixType3>
    void contractRNEADerivatives(
      const DataTpl<Scalar, Options, JointCollectionTpl> & data_full,
      const MimicProjectionTpl<Scalar, Options> & projection,
      const Eigen::MatrixBase<MatrixType1> & rnea_partial_dq,
      const Eigen::MatrixBase<MatrixType2> & rnea_partial_dv,
      const Eigen::MatrixBase<MatrixType3> & rnea_partial_da)
    {
      projection.M_full.template triangularView<Eigen::Upper>() =
        data_full.M.template triangularView<Eigen::Upper>();
      projection.M_full.template triangularView<Eigen::StrictlyLower>() =
        data_full.M.transpose().template triangularView<Eigen::StrictlyLower>();

      mimicContractMatrix(projection, data_full.dtau_dq, rnea_partial_dq);
      mimicContractMatrix(projection, data_full.dtau_dv, rnea_partial_dv);
      mimicContractMatrix(projection, projection.M_full, rnea_partial_da);
    }
  } // namespace internal

  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType1,
    typename TangentVectorType2,
    typename MatrixType1,
    typename MatrixType2,
    typename MatrixType3>
  void computeRNEADerivativesProjected(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_full,
    DataTpl<Scalar, Options, JointCollectionTpl> & data_full,
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType1> & v,
    const Eigen::MatrixBase<TangentVectorType2> & a,
    const Eigen::MatrixBase<MatrixType1> & rnea_partial_dq,
    const Eigen::MatrixBase<MatrixType2> & rnea_partial_dv,
    const Eigen::MatrixBase<MatrixType3> & rnea_partial_da)
  {
    mimicExpandConfiguration(model_full, projection, q, projection.q_full);
    mimicExpandTangent(projection, v, projection.v_full);
    mimicExpandTangent(projection, a, projection.a_full);

    computeRNEADerivatives(
      model_full, data_full, projection.q_full, projection.v_full, projection.a_full);

    internal::contractRNEADerivatives(
      data_full, projection, rnea_partial_dq, rnea_partial_dv, rnea_partial_da);
  }

  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType1,
    typename TangentVectorType2,
    typename MatrixType1,
    typename MatrixType2,
    typename MatrixType3>
  void computeRNEADerivativesProjected(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_full,
    DataTpl<Scalar, Options, JointCollectionTpl> & data_full,
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType1> & v,
    const Eigen::MatrixBase<TangentVectorType2> & a,
    const std::vector<ForceTpl<Scalar, Options>> & fext,
    const Eigen::MatrixBase<MatrixType1> & rnea_partial_dq,
    const Eigen::MatrixBase<MatrixType2> & rnea_partial_dv,
    const Eigen::MatrixBase<MatrixType3> & rnea_partial_da)
  {
    mimicExpandConfiguration(model_full, projection, q, projection.q_full);
    mimicExpandTangent(projection, v, projection.v_full);
    mimicExpandTangent(projection, a, projection.a_full);

    computeRNEADerivatives(
      model_full, data_full, projection.q_full, projection.v_full, projection.a_full, fext);

    internal::contractRNEADerivatives(
      data_full, projection, rnea_partial_dq, rnea_partial_dv, rnea_partial_da);
  }

} // namespace pinocchio
