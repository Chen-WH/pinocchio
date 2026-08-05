//
// Copyright (c) 2026 INRIA
//

#pragma once

// IWYU pragma: begin_keep
#include <Eigen/Core>

#include <cassert>
#include <cstddef>
#include <vector>

#include "pinocchio/macros.hpp"
#include "pinocchio/eigen-common.hpp"
#include "pinocchio/fwd.hpp"

#include "pinocchio/math.hpp"

#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/joint.hpp"

#include "pinocchio/algorithm/rnea-derivatives.hpp"
// IWYU pragma: end_exports

namespace pinocchio
{

  template<typename _Scalar, int _Options>
  struct MimicProjectionTpl;

  template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
  void buildMimicProjection(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_full,
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_reduced,
    MimicProjectionTpl<Scalar, Options> & projection);

  ///
  /// \brief Constant affine coupling map relating a model containing mimic joints (the *reduced*
  ///        model, of dimension \f$ n_r \f$) to the equivalent model where every joint is
  ///        independent (the *expanded* model, of dimension \f$ n_e \f$).
  ///
  /// A mimic joint imposes \f$ q_m = \alpha q_p + \beta \f$ on the expanded coordinates, hence
  /// on the tangent spaces \f$ v_e = G v_r \f$ and \f$ a_e = G a_r \f$ with \f$ G \f$ **constant**
  /// (\f$ \dot G = 0 \f$): there is no acceleration bias term. \f$ G \f$ has exactly one non-zero
  /// entry per row, equal to \f$ 1 \f$ for an independent joint and to \f$ \alpha \f$ for a
  /// mimicking one, which is why it is stored here as an index/scaling pair per row rather than as
  /// a matrix.
  ///
  /// Virtual work then gives \f$ \tau_r = G^\top \tau_e \f$ and, because \f$ G \f$ is constant,
  /// the reduced inverse dynamics derivatives are exactly
  /// \f[
  ///   \frac{\partial \tau_r}{\partial q_r} = G^\top \frac{\partial \tau_e}{\partial q_e} G,
  ///   \quad
  ///   \frac{\partial \tau_r}{\partial v_r} = G^\top \frac{\partial \tau_e}{\partial v_e} G,
  ///   \quad
  ///   M_r = G^\top M_e G.
  /// \f]
  ///
  /// \tparam _Scalar Scalar type of the two models.
  /// \tparam _Options Eigen alignment options of the two models.
  ///
  /// \sa pinocchio::buildMimicProjection, pinocchio::computeRNEADerivativesProjected
  ///
  template<typename _Scalar, int _Options>
  struct MimicProjectionTpl
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    enum
    {
      Options = _Options
    };

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options> VectorXs;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options> MatrixXs;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options | Eigen::RowMajor>
      RowMatrixXs;

    ///
    /// \brief Affine relation carried by a single joint of the reduced model.
    ///
    /// For an independent joint, scaling is \f$ 1 \f$ and offset is \f$ 0 \f$; the joint then
    /// simply copies its configuration from the reduced to the expanded vector.
    ///
    struct JointCoupling
    {
      /// \brief Index of the joint, shared by both models.
      JointIndex joint_id;
      /// \brief Index of the joint configuration in the expanded configuration vector.
      int idx_q_full;
      /// \brief Index of the joint configuration in the reduced configuration vector. For a
      /// mimicking joint, this points at the mimicked joint.
      int idx_q_reduced;
      /// \brief Dimension of the joint configuration, in the expanded model.
      int nq_joint;
      /// \brief Index of the joint tangent space in the expanded tangent vector.
      int idx_v_full;
      /// \brief Index of the joint tangent space in the reduced tangent vector. For a mimicking
      /// joint, this points at the mimicked joint.
      int idx_v_reduced;
      /// \brief Dimension of the joint tangent space, in the expanded model.
      int nv_joint;
      /// \brief Multiplicative factor \f$ \alpha \f$ of the affine relation.
      Scalar scaling;
      /// \brief Additive term \f$ \beta \f$ of the affine relation.
      Scalar offset;
    };

    typedef std::vector<JointCoupling> JointCouplingVector;

    MimicProjectionTpl()
    : nq_full(0)
    , nv_full(0)
    , nq_reduced(0)
    , nv_reduced(0)
    {
    }

    ///
    /// \brief Builds the coupling map relating two structurally identical models, one holding
    ///        mimic joints and the other holding the corresponding independent joints.
    ///
    /// \param[in] model_full Expanded model, without any mimic joint.
    /// \param[in] model_reduced Model holding the mimic joints.
    ///
    template<template<typename, int> class JointCollectionTpl>
    MimicProjectionTpl(
      const ModelTpl<Scalar, Options, JointCollectionTpl> & model_full,
      const ModelTpl<Scalar, Options, JointCollectionTpl> & model_reduced)
    : nq_full(0)
    , nv_full(0)
    , nq_reduced(0)
    , nv_reduced(0)
    {
      buildMimicProjection(model_full, model_reduced, *this);
    }

    /// \brief Dimension of the expanded configuration space \f$ n_{q,e} \f$.
    int nq_full;
    /// \brief Dimension of the expanded tangent space \f$ n_e \f$.
    int nv_full;
    /// \brief Dimension of the reduced configuration space \f$ n_{q,r} \f$.
    int nq_reduced;
    /// \brief Dimension of the reduced tangent space \f$ n_r \f$.
    int nv_reduced;

    /// \brief Per joint affine relation, ordered by joint index, excluding the universe.
    JointCouplingVector joint_couplings;

    /// \brief For each row of the expanded tangent space, the index of the single reduced column
    /// it maps onto. Size nv_full.
    std::vector<int> v_reduced_index;
    /// \brief For each row of the expanded tangent space, the corresponding non-zero entry of
    /// \f$ G \f$. Size nv_full.
    VectorXs v_scaling;

    /// \brief Expanded configuration vector, allocated once. Size nq_full.
    mutable VectorXs q_full;
    /// \brief Expanded velocity vector, allocated once. Size nv_full.
    mutable VectorXs v_full;
    /// \brief Expanded acceleration vector, allocated once. Size nv_full.
    mutable VectorXs a_full;
    /// \brief Symmetrized expanded joint space inertia matrix, stored row-major to match
    /// Data::dtau_dq. Size nv_full x nv_full.
    mutable RowMatrixXs M_full;
    /// \brief Scratch space holding \f$ G^\top D_e \f$ during a contraction.
    /// Size nv_reduced x nv_full.
    mutable MatrixXs contraction_tmp;
  };

  typedef MimicProjectionTpl<context::Scalar, context::Options> MimicProjection;

  ///
  /// \brief Fills a MimicProjectionTpl from an expanded model and its mimicking counterpart.
  ///
  /// Both models must share the same kinematic tree, joint indices and joint types; only the
  /// mimicking joints of \p model_reduced may differ, being JointModelMimicTpl wrapping the
  /// corresponding joint of \p model_full.
  ///
  /// \param[in] model_full Expanded model, without any mimic joint.
  /// \param[in] model_reduced Model holding the mimic joints.
  /// \param[out] projection Coupling map to fill. Its workspace is resized here.
  ///
  template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
  void buildMimicProjection(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_full,
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model_reduced,
    MimicProjectionTpl<Scalar, Options> & projection);

  ///
  /// \brief Returns the dense coupling matrix \f$ G \in \mathbb{R}^{n_e \times n_r} \f$.
  ///
  /// Mostly useful for debugging and testing: every algorithm of this file applies \f$ G \f$
  /// without ever forming it.
  ///
  /// \param[in] projection Coupling map.
  /// \param[out] G Matrix to fill, resized to nv_full x nv_reduced.
  ///
  template<typename Scalar, int Options, typename MatrixType>
  void mimicProjectionMatrix(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<MatrixType> & G);

  ///
  /// \brief Maps a reduced configuration onto the expanded model, \f$ q_e = \gamma(q_r) \f$.
  ///
  /// The joint-wise affine transform is the very one applied internally by JointModelMimicTpl,
  /// so the expanded configuration is consistent with the forward kinematics of the reduced model.
  ///
  /// \param[in] model_full Expanded model.
  /// \param[in] projection Coupling map.
  /// \param[in] q_reduced Configuration of the reduced model. Size nq_reduced.
  /// \param[out] q_full Configuration of the expanded model. Size nq_full.
  ///
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
    const Eigen::MatrixBase<ConfigVectorOut> & q_full);

  ///
  /// \brief Maps a reduced tangent vector onto the expanded model, \f$ x_e = G x_r \f$.
  ///
  /// \param[in] projection Coupling map.
  /// \param[in] x_reduced Tangent vector of the reduced model. Size nv_reduced.
  /// \param[out] x_full Tangent vector of the expanded model. Size nv_full.
  ///
  template<typename Scalar, int Options, typename TangentVectorIn, typename TangentVectorOut>
  void mimicExpandTangent(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<TangentVectorIn> & x_reduced,
    const Eigen::MatrixBase<TangentVectorOut> & x_full);

  ///
  /// \brief Maps an expanded cotangent vector back onto the reduced model,
  ///        \f$ f_r = G^\top f_e \f$.
  ///
  /// This is the virtual work relation turning expanded generalized forces into reduced ones.
  ///
  /// \param[in] projection Coupling map.
  /// \param[in] f_full Cotangent vector of the expanded model. Size nv_full.
  /// \param[out] f_reduced Cotangent vector of the reduced model. Size nv_reduced.
  ///
  template<typename Scalar, int Options, typename VectorIn, typename VectorOut>
  void mimicContractCotangent(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<VectorIn> & f_full,
    const Eigen::MatrixBase<VectorOut> & f_reduced);

  ///
  /// \brief Applies the two-sided contraction \f$ D_r = G^\top D_e G \f$.
  ///
  /// Implemented as two matrix-free passes (a column gather then a row gather) over the single
  /// non-zero entry of each row of \f$ G \f$; \f$ G \f$ is never formed.
  ///
  /// \param[in] projection Coupling map.
  /// \param[in] D_full Expanded matrix. Size nv_full x nv_full.
  /// \param[out] D_reduced Reduced matrix. Size nv_reduced x nv_reduced.
  ///
  template<typename Scalar, int Options, typename MatrixIn, typename MatrixOut>
  void mimicContractMatrix(
    const MimicProjectionTpl<Scalar, Options> & projection,
    const Eigen::MatrixBase<MatrixIn> & D_full,
    const Eigen::MatrixBase<MatrixOut> & D_reduced);

  ///
  /// \brief Computes the inverse dynamics derivatives of a model containing mimic joints, by
  ///        running pinocchio::computeRNEADerivatives on the equivalent expanded model and
  ///        contracting the result with the constant coupling map.
  ///
  /// \param[in] model_full Expanded model, without any mimic joint.
  /// \param[in] data_full Data associated with \p model_full. Overwritten.
  /// \param[in] projection Coupling map built by pinocchio::buildMimicProjection.
  /// \param[in] q Configuration of the reduced model. Size nq_reduced.
  /// \param[in] v Velocity of the reduced model. Size nv_reduced.
  /// \param[in] a Acceleration of the reduced model. Size nv_reduced.
  /// \param[out] rnea_partial_dq Partial derivative of the reduced torque with respect to the
  ///             reduced configuration. Size nv_reduced x nv_reduced.
  /// \param[out] rnea_partial_dv Partial derivative of the reduced torque with respect to the
  ///             reduced velocity. Size nv_reduced x nv_reduced.
  /// \param[out] rnea_partial_da Partial derivative of the reduced torque with respect to the
  ///             reduced acceleration, i.e. the reduced joint space inertia matrix. Size
  ///             nv_reduced x nv_reduced. Unlike pinocchio::computeRNEADerivatives, it is filled
  ///             entirely and not only on its upper triangular part.
  ///
  /// \note The expanded torque is left in data_full.tau; pinocchio::mimicContractCotangent turns
  ///       it into the reduced torque.
  ///
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
    const Eigen::MatrixBase<MatrixType3> & rnea_partial_da);

  ///
  /// \brief Same as above, with external forces applied on the bodies of the expanded model.
  ///
  /// \param[in] fext Vector of external forces expressed in the local frame of each joint. Size
  ///            model_full.njoints.
  ///
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
    const Eigen::MatrixBase<MatrixType3> & rnea_partial_da);

} // namespace pinocchio

// IWYU pragma: begin_exports
#include "pinocchio/src/algorithm/mimic-projection.hxx"
// IWYU pragma: end_exports
