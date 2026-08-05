//
// Copyright (c) 2026 INRIA
//

#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/mimic-projection.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"

#include "utils/model-generator.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_mimic_projection_map)
{
  using namespace Eigen;
  using namespace pinocchio;

  for (int i = 0; i < MimicTestCases::N_CASES; i++)
  {
    const MimicTestCases test_case(i);
    const Model & model_full = test_case.model_full;
    const Model & model_mimic = test_case.model_mimic;
    const MatrixXd & G = test_case.G;

    MimicProjection projection;
    buildMimicProjection(model_full, model_mimic, projection);

    BOOST_CHECK_EQUAL(projection.nq_full, model_full.nq);
    BOOST_CHECK_EQUAL(projection.nv_full, model_full.nv);
    BOOST_CHECK_EQUAL(projection.nq_reduced, model_mimic.nq);
    BOOST_CHECK_EQUAL(projection.nv_reduced, model_mimic.nv);

    // The implicit map must coincide with the reference dense one.
    MatrixXd G_projection;
    mimicProjectionMatrix(projection, G_projection);
    BOOST_CHECK(G_projection.isApprox(G));

    // q_e = gamma(q_r)
    const VectorXd q_mimic = randomConfiguration(model_mimic);
    VectorXd q_full_ref(model_full.nq);
    test_case.toFull(q_mimic, q_full_ref);

    VectorXd q_full(model_full.nq);
    mimicExpandConfiguration(model_full, projection, q_mimic, q_full);
    BOOST_CHECK(q_full.isApprox(q_full_ref));

    // v_e = G v_r
    const VectorXd v_mimic = VectorXd::Random(model_mimic.nv);
    VectorXd v_full(model_full.nv);
    mimicExpandTangent(projection, v_mimic, v_full);
    BOOST_CHECK(v_full.isApprox(G * v_mimic));

    // tau_r = G^T tau_e
    const VectorXd tau_full = VectorXd::Random(model_full.nv);
    VectorXd tau_mimic(model_mimic.nv);
    mimicContractCotangent(projection, tau_full, tau_mimic);
    BOOST_CHECK(tau_mimic.isApprox(G.transpose() * tau_full));

    // D_r = G^T D_e G
    const MatrixXd D_full = MatrixXd::Random(model_full.nv, model_full.nv);
    MatrixXd D_mimic(model_mimic.nv, model_mimic.nv);
    mimicContractMatrix(projection, D_full, D_mimic);
    BOOST_CHECK(D_mimic.isApprox(G.transpose() * D_full * G));
  }
}

BOOST_AUTO_TEST_CASE(test_mimic_projection_rnea)
{
  using namespace Eigen;
  using namespace pinocchio;

  for (int i = 0; i < MimicTestCases::N_CASES; i++)
  {
    const MimicTestCases test_case(i);
    const Model & model_full = test_case.model_full;
    const Model & model_mimic = test_case.model_mimic;

    const MimicProjection projection(model_full, model_mimic);

    const VectorXd q = randomConfiguration(model_mimic);
    const VectorXd v = VectorXd::Random(model_mimic.nv);
    const VectorXd a = VectorXd::Random(model_mimic.nv);

    // The expanded inverse dynamics, contracted back, is the mimic inverse dynamics.
    Data data_full(model_full);
    VectorXd q_full(model_full.nq), v_full(model_full.nv), a_full(model_full.nv);
    mimicExpandConfiguration(model_full, projection, q, q_full);
    mimicExpandTangent(projection, v, v_full);
    mimicExpandTangent(projection, a, a_full);
    rnea(model_full, data_full, q_full, v_full, a_full);

    VectorXd tau_projected(model_mimic.nv);
    mimicContractCotangent(projection, data_full.tau, tau_projected);

    Data data_mimic(model_mimic);
    rnea(model_mimic, data_mimic, q, v, a);

    BOOST_CHECK(tau_projected.isApprox(data_mimic.tau));
  }
}

BOOST_AUTO_TEST_CASE(test_mimic_projection_rnea_derivatives)
{
  using namespace Eigen;
  using namespace pinocchio;

  for (int i = 0; i < MimicTestCases::N_CASES; i++)
  {
    const MimicTestCases test_case(i);
    const Model & model_full = test_case.model_full;
    const Model & model_mimic = test_case.model_mimic;
    const MatrixXd & G = test_case.G;

    const MimicProjection projection(model_full, model_mimic);

    const VectorXd q = randomConfiguration(model_mimic);
    const VectorXd v = VectorXd::Random(model_mimic.nv);
    const VectorXd a = VectorXd::Random(model_mimic.nv);

    Data data(model_full);
    MatrixXd dtau_dq(model_mimic.nv, model_mimic.nv);
    MatrixXd dtau_dv(model_mimic.nv, model_mimic.nv);
    MatrixXd dtau_da(model_mimic.nv, model_mimic.nv);
    computeRNEADerivativesProjected(
      model_full, data, projection, q, v, a, dtau_dq, dtau_dv, dtau_da);

    // Reference: the same contraction, done with the dense G.
    VectorXd q_full(model_full.nq);
    test_case.toFull(q, q_full);
    const VectorXd v_full = G * v;
    const VectorXd a_full = G * a;

    Data data_ref(model_full);
    computeRNEADerivatives(model_full, data_ref, q_full, v_full, a_full);
    data_ref.M.triangularView<StrictlyLower>() =
      data_ref.M.transpose().triangularView<StrictlyLower>();

    BOOST_CHECK(dtau_dq.isApprox(G.transpose() * data_ref.dtau_dq * G));
    BOOST_CHECK(dtau_dv.isApprox(G.transpose() * data_ref.dtau_dv * G));
    BOOST_CHECK(dtau_da.isApprox(G.transpose() * data_ref.M * G));

    // The reduced mass matrix must match the one the native mimic CRBA computes.
    Data data_crba(model_mimic);
    crba(model_mimic, data_crba, q, Convention::WORLD);
    data_crba.M.triangularView<StrictlyLower>() =
      data_crba.M.transpose().triangularView<StrictlyLower>();
    BOOST_CHECK(dtau_da.isApprox(data_crba.M));
  }
}

BOOST_AUTO_TEST_CASE(test_mimic_projection_rnea_derivatives_fd)
{
  using namespace Eigen;
  using namespace pinocchio;

  const double alpha = 1e-8;

  for (int i = 0; i < MimicTestCases::N_CASES; i++)
  {
    const MimicTestCases test_case(i);
    const Model & model_full = test_case.model_full;
    const Model & model_mimic = test_case.model_mimic;
    const Eigen::DenseIndex nv = model_mimic.nv;

    const MimicProjection projection(model_full, model_mimic);

    const VectorXd q = randomConfiguration(model_mimic);
    const VectorXd v = VectorXd::Random(nv);
    const VectorXd a = VectorXd::Random(nv);

    Data data(model_full);
    MatrixXd dtau_dq(nv, nv), dtau_dv(nv, nv), dtau_da(nv, nv);
    computeRNEADerivativesProjected(
      model_full, data, projection, q, v, a, dtau_dq, dtau_dv, dtau_da);

    // Finite differences on the native mimic inverse dynamics, which is an oracle fully
    // independent from the expanded model.
    Data data_fd(model_mimic), data_ref(model_mimic);
    const VectorXd tau0 = rnea(model_mimic, data_ref, q, v, a);

    MatrixXd dtau_dq_fd(nv, nv);
    VectorXd dq(VectorXd::Zero(nv));
    for (Eigen::DenseIndex k = 0; k < nv; ++k)
    {
      dq[k] = alpha;
      const VectorXd q_plus = integrate(model_mimic, q, dq);
      dtau_dq_fd.col(k) = (rnea(model_mimic, data_fd, q_plus, v, a) - tau0) / alpha;
      dq[k] = 0.;
    }
    BOOST_CHECK(dtau_dq.isApprox(dtau_dq_fd, std::sqrt(alpha)));

    MatrixXd dtau_dv_fd(nv, nv);
    VectorXd v_plus(v);
    for (Eigen::DenseIndex k = 0; k < nv; ++k)
    {
      v_plus[k] += alpha;
      dtau_dv_fd.col(k) = (rnea(model_mimic, data_fd, q, v_plus, a) - tau0) / alpha;
      v_plus[k] = v[k];
    }
    BOOST_CHECK(dtau_dv.isApprox(dtau_dv_fd, std::sqrt(alpha)));

    MatrixXd dtau_da_fd(nv, nv);
    VectorXd a_plus(a);
    for (Eigen::DenseIndex k = 0; k < nv; ++k)
    {
      a_plus[k] += alpha;
      dtau_da_fd.col(k) = (rnea(model_mimic, data_fd, q, v, a_plus) - tau0) / alpha;
      a_plus[k] = a[k];
    }
    BOOST_CHECK(dtau_da.isApprox(dtau_da_fd, std::sqrt(alpha)));
  }
}

BOOST_AUTO_TEST_CASE(test_mimic_projection_rnea_derivatives_fext)
{
  using namespace Eigen;
  using namespace pinocchio;

  for (int i = 0; i < MimicTestCases::N_CASES; i++)
  {
    const MimicTestCases test_case(i);
    const Model & model_full = test_case.model_full;
    const Model & model_mimic = test_case.model_mimic;
    const MatrixXd & G = test_case.G;

    const MimicProjection projection(model_full, model_mimic);

    const VectorXd q = randomConfiguration(model_mimic);
    const VectorXd v = VectorXd::Random(model_mimic.nv);
    const VectorXd a = VectorXd::Random(model_mimic.nv);

    typedef std::vector<Force> ForceVector;
    ForceVector fext((size_t)model_full.njoints);
    for (ForceVector::iterator it = fext.begin(); it != fext.end(); ++it)
      (*it).setRandom();

    Data data(model_full);
    MatrixXd dtau_dq(model_mimic.nv, model_mimic.nv);
    MatrixXd dtau_dv(model_mimic.nv, model_mimic.nv);
    MatrixXd dtau_da(model_mimic.nv, model_mimic.nv);
    computeRNEADerivativesProjected(
      model_full, data, projection, q, v, a, fext, dtau_dq, dtau_dv, dtau_da);

    VectorXd q_full(model_full.nq);
    test_case.toFull(q, q_full);
    const VectorXd v_full = G * v;
    const VectorXd a_full = G * a;

    Data data_ref(model_full);
    computeRNEADerivatives(model_full, data_ref, q_full, v_full, a_full, fext);
    data_ref.M.triangularView<StrictlyLower>() =
      data_ref.M.transpose().triangularView<StrictlyLower>();

    BOOST_CHECK(dtau_dq.isApprox(G.transpose() * data_ref.dtau_dq * G));
    BOOST_CHECK(dtau_dv.isApprox(G.transpose() * data_ref.dtau_dv * G));
    BOOST_CHECK(dtau_da.isApprox(G.transpose() * data_ref.M * G));
  }
}

BOOST_AUTO_TEST_SUITE_END()
