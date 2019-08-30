#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Geometry>
#include <boost/format.hpp>
#include <iostream>
#include <tesseract_kinematics/core/utils.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/kinematic_terms.hpp>
#include <trajopt/utils.hpp>
#include <trajopt_sco/expr_ops.hpp>
#include <trajopt_sco/modeling_utils.hpp>
#include <trajopt_utils/eigen_conversions.hpp>
#include <trajopt_utils/eigen_slicing.hpp>
#include <trajopt_utils/logging.hpp>
#include <trajopt_utils/stl_to_string.hpp>

using namespace std;
using namespace sco;
using namespace Eigen;
using namespace util;

namespace
{
#if 0
Vector3d rotVec(const Matrix3d& m) {
  Quaterniond q; q = m;
  return Vector3d(q.x(), q.y(), q.z());
}
#endif

#if 0
VectorXd concat(const VectorXd& a, const VectorXd& b) {
  VectorXd out(a.size()+b.size());
  out.topRows(a.size()) = a;
  out.middleRows(a.size(), b.size()) = b;
  return out;
}

template <typename T>
vector<T> concat(const vector<T>& a, const vector<T>& b) {
  vector<T> out;
  vector<int> x;
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return out;
}
#endif
}  // namespace

namespace trajopt
{
VectorXd DynamicCartPoseErrCalculator::operator()(const VectorXd& dof_vals) const
{
  Isometry3d new_pose, target_pose;
  manip_->calcFwdKin(new_pose, dof_vals, kin_link_->link_name);
  manip_->calcFwdKin(target_pose, dof_vals, kin_target_->link_name);

  Eigen::Isometry3d link_tf = world_to_base_ * new_pose * kin_link_->transform * tcp_;
  Eigen::Isometry3d target_tf = world_to_base_ * target_pose * kin_target_->transform;

  Isometry3d pose_err = target_tf.inverse() * link_tf;
  Quaterniond q(pose_err.rotation());
  VectorXd err = concat(Vector3d(q.x(), q.y(), q.z()), pose_err.translation());
  return err;
}

void DynamicCartPoseErrCalculator::Plot(const tesseract_visualization::Visualization::Ptr& plotter,
                                        const VectorXd& dof_vals)
{
  Isometry3d cur_pose, target_pose;

  manip_->calcFwdKin(cur_pose, dof_vals, kin_link_->link_name);
  manip_->calcFwdKin(target_pose, dof_vals, kin_target_->link_name);

  Eigen::Isometry3d cur_tf = world_to_base_ * cur_pose * kin_link_->transform * tcp_;
  Eigen::Isometry3d target_tf = world_to_base_ * target_pose * kin_target_->transform;

  plotter->plotAxis(cur_tf, 0.05);
  plotter->plotAxis(target_tf, 0.05);
  plotter->plotArrow(cur_tf.translation(), target_tf.translation(), Eigen::Vector4d(1, 0, 1, 1), 0.005);
}

VectorXd CartPoseErrCalculator::operator()(const VectorXd& dof_vals) const
{
  Isometry3d new_pose;
  manip_->calcFwdKin(new_pose, dof_vals, kin_link_->link_name);

  new_pose = world_to_base_ * new_pose * kin_link_->transform * tcp_;

  Isometry3d pose_err = pose_inv_ * new_pose;
  Quaterniond q(pose_err.rotation());
  VectorXd err = concat(Vector3d(q.x(), q.y(), q.z()), pose_err.translation());
  return err;
}

void CartPoseErrCalculator::Plot(const tesseract_visualization::Visualization::Ptr& plotter, const VectorXd& dof_vals)
{
  Isometry3d cur_pose;
  manip_->calcFwdKin(cur_pose, dof_vals, kin_link_->link_name);

  cur_pose = world_to_base_ * cur_pose * kin_link_->transform * tcp_;

  Isometry3d target = pose_inv_.inverse();

  plotter->plotAxis(cur_pose, 0.05);
  plotter->plotAxis(target, 0.05);
  plotter->plotArrow(cur_pose.translation(), target.translation(), Eigen::Vector4d(1, 0, 1, 1), 0.005);
}

MatrixXd CartVelJacCalculator::operator()(const VectorXd& dof_vals) const
{
  int n_dof = static_cast<int>(manip_->numJoints());
  MatrixXd out(6, 2 * n_dof);

  MatrixXd jac0, jac1;
  Eigen::Isometry3d tf0, tf1;

  jac0.resize(6, manip_->numJoints());
  jac1.resize(6, manip_->numJoints());

  if (tcp_.translation().isZero())
  {
    manip_->calcFwdKin(tf0, dof_vals.topRows(n_dof), kin_link_->link_name);
    manip_->calcJacobian(jac0, dof_vals.topRows(n_dof), kin_link_->link_name);
    tesseract_kinematics::jacobianChangeBase(jac0, world_to_base_);
    tesseract_kinematics::jacobianChangeRefPoint(jac0,
                                                 (world_to_base_ * tf0).linear() * kin_link_->transform.translation());

    manip_->calcFwdKin(tf1, dof_vals.bottomRows(n_dof), kin_link_->link_name);
    manip_->calcJacobian(jac1, dof_vals.bottomRows(n_dof), kin_link_->link_name);
    tesseract_kinematics::jacobianChangeBase(jac1, world_to_base_);
    tesseract_kinematics::jacobianChangeRefPoint(jac1,
                                                 (world_to_base_ * tf1).linear() * kin_link_->transform.translation());
  }
  else
  {
    manip_->calcFwdKin(tf0, dof_vals.topRows(n_dof), kin_link_->link_name);
    manip_->calcJacobian(jac0, dof_vals.topRows(n_dof), kin_link_->link_name);
    tesseract_kinematics::jacobianChangeBase(jac0, world_to_base_);
    tesseract_kinematics::jacobianChangeRefPoint(
        jac0, (world_to_base_ * tf0).linear() * (kin_link_->transform * tcp_).translation());

    manip_->calcFwdKin(tf1, dof_vals.bottomRows(n_dof), kin_link_->link_name);
    manip_->calcJacobian(jac1, dof_vals.bottomRows(n_dof), kin_link_->link_name);
    tesseract_kinematics::jacobianChangeBase(jac1, world_to_base_);
    tesseract_kinematics::jacobianChangeRefPoint(
        jac1, (world_to_base_ * tf1).linear() * (kin_link_->transform * tcp_).translation());
  }

  out.block(0, 0, 3, n_dof) = -jac0.topRows(3);
  out.block(0, n_dof, 3, n_dof) = jac1.topRows(3);
  out.block(3, 0, 3, n_dof) = jac0.topRows(3);
  out.block(3, n_dof, 3, n_dof) = -jac1.topRows(3);
  return out;
}

VectorXd CartVelErrCalculator::operator()(const VectorXd& dof_vals) const
{
  int n_dof = static_cast<int>(manip_->numJoints());
  Isometry3d pose0, pose1;

  manip_->calcFwdKin(pose0, dof_vals.topRows(n_dof), kin_link_->link_name);
  manip_->calcFwdKin(pose1, dof_vals.bottomRows(n_dof), kin_link_->link_name);

  pose0 = world_to_base_ * pose0 * kin_link_->transform * tcp_;
  pose1 = world_to_base_ * pose1 * kin_link_->transform * tcp_;

  VectorXd out(6);
  out.topRows(3) = (pose1.translation() - pose0.translation() - Vector3d(limit_, limit_, limit_));
  out.bottomRows(3) = (pose0.translation() - pose1.translation() - Vector3d(limit_, limit_, limit_));
  return out;
}

namespace
{
Eigen::VectorXd VarDtDiff(const VectorXd& var_vals)
{
  // (x1-x0)*(1/dt)
  assert(var_vals.rows() % 2 == 0);
  const int half = static_cast<int>(var_vals.rows()) / 2;
  const int num_diff = half - 1;

  const auto x2 = var_vals.segment(1, num_diff);
  const auto x1 = var_vals.segment(0, num_diff);
  const auto dt_inv = var_vals.segment(half + 1, num_diff);

  VectorXd result(num_diff * 2);
  result.topRows(num_diff) = (x2 - x1).array() * dt_inv.array();
  result.bottomRows(num_diff) = dt_inv;

  return result;
}

Eigen::VectorXd VarDtErr(const VectorXd& vals_and_dts, double target, double upper_tol, double lower_tol)
{
  assert(vals_and_dts.rows() % 2 == 0);
  const int half = static_cast<int>(vals_and_dts.rows()) / 2;
  const auto vals = vals_and_dts.topRows(half);

  // Note that for equality terms tols are 0, so error is effectively doubled
  VectorXd result(vals.rows() * 2);
  result.topRows(vals.rows()) = -(upper_tol - (vals.array() - target));
  result.bottomRows(vals.rows()) = lower_tol - (vals.array() - target);

  return result;
}
}  // namespace

Eigen::VectorXd JointVelErrCalculator::operator()(const VectorXd& var_vals) const
{
  const VectorXd vel = VarDtDiff(var_vals);
  return VarDtErr(vel, target_, upper_tol_, lower_tol_);
}

MatrixXd JointVelJacCalculator::operator()(const VectorXd& var_vals) const
{
  // var_vals = (theta_t1, theta_t2, theta_t3 ... 1/dt_1, 1/dt_2, 1/dt_3 ...)
  const int num_vals = static_cast<int>(var_vals.rows());
  const int half = num_vals / 2;
  const int num_vels = half - 1;
  MatrixXd jac = MatrixXd::Zero(num_vels * 2, num_vals);

  for (int i = 0; i < num_vels; i++)
  {
    // v = (j_i+1 - j_i)*(1/dt)
    // We calculate v with the dt from the second pt
    int time_index = i + half + 1;
    // dv_i/dj_i = -(1/dt)
    jac(i, i) = -1.0 * var_vals(time_index);
    // dv_i/dj_i+1 = (1/dt)
    jac(i, i + 1) = 1.0 * var_vals(time_index);
    // dv_i/dt_i = j_i+1 - j_i
    jac(i, time_index) = var_vals(i + 1) - var_vals(i);
    // All others are 0
  }

  // bottom half is negative velocities
  jac.bottomRows(num_vels) = -jac.topRows(num_vels);

  return jac;
}

// TODO: convert to (1/dt) and use central finite difference method
VectorXd JointAccErrCalculator::operator()(const VectorXd& var_vals) const
{
  const VectorXd vel = VarDtDiff(var_vals);
  const VectorXd acc = VarDtDiff(vel);
  return VarDtErr(acc, target_, upper_tol_, lower_tol_);
}

MatrixXd JointAccJacCalculator::operator()(const VectorXd& var_vals) const
{
  const int num_vals = static_cast<int>(var_vals.rows());
  const int half = num_vals / 2;
  const int num_accs = half - 2;
  MatrixXd jac = MatrixXd::Zero(num_accs * 2, num_vals);

  for (int i = 0; i < num_accs; i++)
  {
    const double dt1 = var_vals(i + half + 1);
    const double dt2 = var_vals(i + half + 2);
    const double x0 = var_vals(i);
    const double x1 = var_vals(i + 1);
    const double x2 = var_vals(i + 2);

    jac(i, i) = dt1 * dt2;
    jac(i, i + 1) = -dt2 * (dt1 + dt2);
    jac(i, i + 2) = dt2 * dt2;
    jac(i, i + half + 1) = dt2 * (x0 - x1);
    jac(i, i + half + 2) = dt1 * (x0 - x1) + 2 * dt2 * (x2 - x1);
  }

  return jac;
}

// TODO: convert to (1/dt) and use central finite difference method
VectorXd JointJerkErrCalculator::operator()(const VectorXd& var_vals) const
{
  const VectorXd vel = VarDtDiff(var_vals);
  const VectorXd acc = VarDtDiff(vel);
  const VectorXd jerk = VarDtDiff(acc);
  return VarDtErr(jerk, target_, upper_tol_, lower_tol_);
}

MatrixXd JointJerkJacCalculator::operator()(const VectorXd& var_vals) const
{
  const int num_vals = static_cast<int>(var_vals.rows());
  const int half = num_vals / 2;
  const int num_jerks = half - 3;
  MatrixXd jac = MatrixXd::Zero(num_jerks * 2, num_vals);

  for (int i = 0; i < num_jerks; i++)
  {
    const double dt1 = var_vals(i + half + 1);
    const double dt2 = var_vals(i + half + 2);
    const double dt3 = var_vals(i + half + 3);
    const double x0 = var_vals(i);
    const double x1 = var_vals(i + 1);
    const double x2 = var_vals(i + 2);
    const double x3 = var_vals(i + 3);

    jac(i, i) = -dt1 * -dt2 * -dt3;
    jac(i, i + 1) = dt3 * (dt2 * dt3 + dt2 * (dt1 + dt2));
    jac(i, i + 2) = -dt3 * (dt2 * dt2 + dt3 * (dt2 + dt3));
    jac(i, i + 3) = dt3 * dt3 * dt3;
    jac(i, i + half + 1) = dt2 * dt3 * (x1 - x0);
    jac(i, i + half + 2) = dt3 * (dt1 * (x1 - x0) - 2 * dt2 * (x2 - x1) + dt3 * (x1 - x2));
    jac(i, i + half + 3) = -dt2 * (-dt1 * (-x0 + x1) + dt2 * (-x1 + x2)) +
                           dt3 * (-dt2 * (-x1 + x2) + dt3 * (-x2 + x3)) +
                           dt3 * (-dt2 * (-x1 + x2) + 2 * dt3 * (-x2 + x3));
  }

  return jac;
}

namespace
{
Eigen::VectorXd FixedDtDiff(const VectorXd& var_vals, double dt)
{
  const int num_diff = static_cast<int>(var_vals.rows()) - 1;
  const auto x2 = var_vals.segment(1, num_diff);
  const auto x1 = var_vals.segment(0, num_diff);
  return (x2 - x1) / dt;
}

Eigen::VectorXd FixedDtErr(const VectorXd& vals, double target, double upper_tol, double lower_tol)
{
  // Note that for equality terms tols are 0, so error is effectively doubled
  VectorXd result(vals.rows() * 2);
  result.topRows(vals.rows()) = -(upper_tol - (vals.array() - target));
  result.bottomRows(vals.rows()) = lower_tol - (vals.array() - target);

  return result;
}
}  // namespace

Eigen::VectorXd JointVelErrCalculatorFixedDt::operator()(const VectorXd& var_vals) const
{
  const VectorXd vel = FixedDtDiff(var_vals, dt_);
  return FixedDtErr(vel, target_, upper_tol_, lower_tol_);
}

MatrixXd JointVelJacCalculatorFixedDt::operator()(const VectorXd& var_vals) const
{
  const int num_vals = static_cast<int>(var_vals.rows());
  const int num_vels = num_vals - 1;
  MatrixXd jac = MatrixXd::Zero(num_vels * 2, num_vals);

  for (int i = 0; i < num_vels; i++)
  {
    jac(i, i) = -1.0 * dt_;
    jac(i, i + 1) = 1.0 * dt_;
    // All others are 0
  }

  // bottom half is negative velocities
  jac.bottomRows(num_vels) = -jac.topRows(num_vels);

  return jac;
}

VectorXd JointAccErrCalculatorFixedDt::operator()(const VectorXd& var_vals) const
{
  const VectorXd vel = FixedDtDiff(var_vals, dt_);
  const VectorXd acc = FixedDtDiff(vel, dt_);
  return FixedDtErr(acc, target_, upper_tol_, lower_tol_);
}

MatrixXd JointAccJacCalculatorFixedDt::operator()(const VectorXd& var_vals) const
{
  const int num_vals = static_cast<int>(var_vals.rows());
  const int num_accs = num_vals - 2;
  MatrixXd jac = MatrixXd::Zero(num_accs * 2, num_vals);

  const double dt_sq = dt_ * dt_;

  for (int i = 0; i < num_accs; i++)
  {
    jac(i, i) = dt_sq;
    jac(i, i + 1) = -2.0 * dt_sq;
    jac(i, i + 2) = dt_sq;
  }

  return jac;
}

VectorXd JointJerkErrCalculatorFixedDt::operator()(const VectorXd& var_vals) const
{
  const VectorXd vel = FixedDtDiff(var_vals, dt_);
  const VectorXd acc = FixedDtDiff(vel, dt_);
  const VectorXd jerk = FixedDtDiff(acc, dt_);
  return FixedDtErr(jerk, target_, upper_tol_, lower_tol_);
}

MatrixXd JointJerkJacCalculatorFixedDt::operator()(const VectorXd& var_vals) const
{
  const int num_vals = static_cast<int>(var_vals.rows());
  const int num_jerks = num_vals - 3;
  MatrixXd jac = MatrixXd::Zero(num_jerks * 2, num_vals);

  const double dt_cb = dt_ * dt_ * dt_;

  for (int i = 0; i < num_jerks; i++)
  {
    jac(i, i) = -dt_cb;
    jac(i, i + 1) = 3.0 * dt_cb;
    jac(i, i + 2) = -3.0 * dt_cb;
    jac(i, i + 3) = dt_cb;
  }

  return jac;
}

VectorXd TimeCostCalculator::operator()(const VectorXd& time_vals) const
{
  VectorXd total(1);
  total(0) = time_vals.cwiseInverse().sum() - limit_;
  return total;
}

MatrixXd TimeCostJacCalculator::operator()(const VectorXd& time_vals) const
{
  MatrixXd jac(1, time_vals.rows());
  jac.row(0) = -1 * time_vals.cwiseAbs2().cwiseInverse();
  return jac;
}

}  // namespace trajopt
