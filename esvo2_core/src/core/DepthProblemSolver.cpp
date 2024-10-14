#include <esvo2_core/core/DepthProblemSolver.h>
#include <tbb/parallel_for.h>
// #include <tbb/global_control.h>
#include <thread>
#include <functional>
#include <fstream>
#include <esvo2_core/tools/TicToc.h>
//#define DEPTH_PROBLEM_SOLVER_LOG
namespace esvo2_core
{
namespace core
{
DepthProblemSolver::DepthProblemSolver(
  CameraSystem::Ptr & camSysPtr,
  std::shared_ptr<DepthProblemConfig> & dpConfigPtr,
  DepthProblemType dpType,
  size_t numThread,
  bool slove_lr )
  :
  camSysPtr_(camSysPtr),
  dpConfigPtr_(dpConfigPtr),
  dpType_(dpType),
  NUM_THREAD_(numThread),
  slove_lr_(slove_lr)
{
  dpConfigPtr_->patchSize_X_ = (dpConfigPtr_->patchSize_X_ + 1) / 2;
  dpConfigPtr_->patchSize_Y_ = (dpConfigPtr_->patchSize_Y_ + 1) / 2;

  if(dpConfigPtr_->patchSize_X_ % 2 == 0)
    dpConfigPtr_->patchSize_X_++;
  
  if(dpConfigPtr_->patchSize_Y_ % 2 == 0)
    dpConfigPtr_->patchSize_Y_++;
}

DepthProblemSolver::~DepthProblemSolver()
{
}

void DepthProblemSolver::solve(
  std::vector<EventMatchPair>* pvEMP,
  constStampedTimeSurfaceObs* pStampedTsObs,
  std::vector<DepthPoint> &vdp )
{
//  TicToc tt;
//  tt.tic();
  //distribute the loads
  // NUM_THREAD_ = 1;
  std::vector<Job> jobs(NUM_THREAD_);
  for(size_t i = 0;i < NUM_THREAD_;i++)
  {
    jobs[i].i_thread_ = i;
    jobs[i].pvEMP_ = pvEMP;
    jobs[i].pStamped_TS_obs_ = pStampedTsObs;
    // if(dpType_ == NUMERICAL)
    // {
    //   jobs[i].numDiff_dProblemPtr_ = std::make_shared<Eigen::NumericalDiff<DepthProblem> >(dpConfigPtr_, camSysPtr_);
    jobs[i].dProblemPtr_ = std::make_shared<DepthProblem>(dpConfigPtr_, camSysPtr_); //NOTE: The ANALYTICAL version is not provided in this version.
    // }
      
    // else if(dpType_ == ANALYTICAL)
    //   jobs[i].dProblemPtr_ = std::make_shared<DepthProblem>(dpConfigPtr_, camSysPtr_); //NOTE: The ANALYTICAL version is not provided in this version.
    // else
    // {
    //   LOG(ERROR) << "Wrong Depth Problem Type is assigned!!!";
    //   exit(-1);
    // }

    jobs[i].vdpPtr_ = std::make_shared<std::vector<DepthPoint> >();
  }
//  LOG(INFO) << "(DepthProblemSolver) distribute the loads: " << tt.toc() << " ms.";
// //  tt.tic();
//   //
  std::vector<std::thread> threads;
  for(size_t i = 0; i < NUM_THREAD_;i++)
    threads.emplace_back(std::bind(&DepthProblemSolver::init_single_point, this, jobs[i]));
  for( auto & thread : threads)
  {
    if(thread.joinable())
      thread.join();
  }
//   //

  // 使用TBB并行计算任务处理结构体数组
  // tbb::global_control c(tbb::global_control::max_allowed_parallelism, NUM_THREAD_);
  // tbb::parallel_for(tbb::blocked_range<size_t>(0, jobs.size()),
  //                   [&](const tbb::blocked_range<size_t>& range) {
  //                       // 对于每个范围内的索引，调用结构体操作函数
  //                       for (size_t i = range.begin(); i < range.end(); ++i) {
  //                           DepthProblemSolver::solve_multiple_problems(jobs[i]);
  //                       }
  //                   });
  vdp.clear();
  size_t numPoints = 0;
  for(size_t i = 0;i < NUM_THREAD_;i++)
  {
#ifdef DEPTH_PROBLEM_SOLVER_LOG
    LOG(INFO) << "The " << i << " thread reconstructs " << jobs[i].vdpPtr_->size() << " points";
#endif
    numPoints += jobs[i].vdpPtr_->size();
  }
  vdp.reserve(numPoints);
  for(size_t i = 0;i < NUM_THREAD_;i++)
    vdp.insert(vdp.end(), jobs[i].vdpPtr_->begin(), jobs[i].vdpPtr_->end());
//  LOG(INFO) << "(DepthProblemSolver) copy results: " << tt.toc() << " ms.";

  // vdp.clear();
  // vdp.reserve(pvEMP->size());
  // Eigen::Matrix<double, 4, 4> T_world_virtual = pStampedTsObs->second.tr_.getTransformationMatrix();
  // #pragma omp parallel for num_threads(4)
  // for(size_t i = 0; i < pvEMP->size(); i++)
  // {
    
  //   Eigen::Vector2d coor = (*pvEMP)[i].x_left_;
  //   std::shared_ptr<DepthProblem> dProblemPtr_ = std::make_shared<DepthProblem>(dpConfigPtr_, camSysPtr_);
  //   dProblemPtr_->setProblem(coor);
  //   init_single_point((*pvEMP)[i].invDepth_, dProblemPtr_, (*pvEMP)[i], T_world_virtual,vdp);
  // }
}

void DepthProblemSolver::solve_multiple_problems(Job & job)
{
  size_t i_thread = job.i_thread_;
  size_t numEvent = job.pvEMP_->size();
  job.vdpPtr_->clear();
  job.vdpPtr_->reserve(numEvent / NUM_THREAD_ + 1);

  constStampedTimeSurfaceObs* pStampedTsObs = job.pStamped_TS_obs_;

  // loop through vdp and call solve_single_problem
  for(size_t i = i_thread; i < numEvent; i+=NUM_THREAD_)
  {
    Eigen::Vector2d coor = (*job.pvEMP_)[i].x_left_;
    // Eigen::Matrix<double, 4, 4> T_world_virtual = (*job.pvEMP_)[i].trans_.getTransformationMatrix();
    Eigen::Matrix<double, 4, 4> T_world_virtual = pStampedTsObs->second.tr_.getTransformationMatrix();
    double d_init = (*job.pvEMP_)[i].invDepth_;

    double result[3], result_ref[3];
    bool bProblemSolved = false;
    if(dpType_ == NUMERICAL)
    {
      job.dProblemPtr_->setProblem(coor, T_world_virtual, pStampedTsObs, slove_lr_);
      // init_single_point(d_init,job.dProblemPtr_, result_ref);
      job.numDiff_dProblemPtr_->setProblem(coor, T_world_virtual, pStampedTsObs, slove_lr_);
      bProblemSolved = solve_single_problem_numerical(d_init,job.numDiff_dProblemPtr_, result);
    }
    else if(dpType_ == ANALYTICAL)
    {
      job.dProblemPtr_->setProblem(coor, T_world_virtual, pStampedTsObs, slove_lr_);
      bProblemSolved = solve_single_problem_analytical(d_init,job.dProblemPtr_, result);
    }
    else
    {
      LOG(ERROR) << "Wrong Depth Problem Type is assigned!!!";
      exit(-1);
    }

    if(bProblemSolved)
    {
      DepthPoint dp(std::floor(coor(1)), std::floor(coor(0)));
      dp.update_x(coor);
      Eigen::Vector3d p_cam;
      camSysPtr_->cam_left_ptr_->cam2World(coor, result[0], p_cam);
      dp.update_p_cam(p_cam);
      if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0 || strcmp(dpConfigPtr_->LSnorm_.c_str(), "zncc") == 0)
        dp.update(result[0], result[1]);
      else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
      {
        double scale2_rho = result[1] * (dpConfigPtr_->td_nu_ - 2) / dpConfigPtr_->td_nu_;
        dp.update_studentT(result[0], scale2_rho, result[1], dpConfigPtr_->td_nu_);
      }
      else
        exit(-1);

      dp.residual() = result[2];
      // dp.residual() = 1;
      dp.updatePose(T_world_virtual);
      job.vdpPtr_->push_back(dp);
    }
  }
}




bool DepthProblemSolver::init_single_point(
  Job & job)
  {
    size_t i_thread = job.i_thread_;
    size_t numEvent = job.pvEMP_->size();
    job.vdpPtr_->clear();
    job.vdpPtr_->reserve(numEvent / NUM_THREAD_ + 1);

    constStampedTimeSurfaceObs* pStampedTsObs = job.pStamped_TS_obs_;
    // TicToc t, t_in;
    // double t_in_all = 0;
    Eigen::Matrix<double, 4, 4> T_world_virtual = pStampedTsObs->second.tr_.getTransformationMatrix();
    for(size_t i = i_thread; i < numEvent; i+=NUM_THREAD_)
    {
      Eigen::Vector2d coor = (*job.pvEMP_)[i].x_left_;
      double d_init = (*job.pvEMP_)[i].invDepth_;
      if(i == i_thread)
        job.dProblemPtr_->setProblem(coor, T_world_virtual, pStampedTsObs, slove_lr_);
      else
        job.dProblemPtr_->setProblem(coor);
    
      Eigen::VectorXd x(1);
      x << d_init;
      float result[3];

      int wx, wy;
      wx = job.dProblemPtr_->dpConfigPtr_->patchSize_X_;
      wy = job.dProblemPtr_->dpConfigPtr_->patchSize_Y_;


      Eigen::VectorXd val1(wx*wy), val2(wx*wy), fjac(wx*wy);
      int numValid = (*job.dProblemPtr_)(x, val1);
      x(0, 0) = x(0, 0) + d_init * 0.05;

      // t_in.tic();
      numValid = (*job.dProblemPtr_)(x, val2);
      // t_in_all += t_in.toc();
      fjac = (val2 - val1) / (d_init * 0.05);

      // Eigen::HouseholderQR<Eigen::MatrixXd> qrfac(fjac);
      // Eigen::MatrixXd R = qrfac.matrixQR().triangularView<Eigen::Upper>();
      double R = fjac.norm();

      if(R < 0.1 && R > -0.1)
        R = 0.1;

      
      result[0] = d_init;
      result[1] = std::pow(job.dProblemPtr_->dpConfigPtr_->td_stdvar_,2) / (R*R);
      result[2] = val1.norm() * val1.norm();

      DepthPoint dp(std::floor(coor(1)), std::floor(coor(0)));
      dp.update_x(coor);
      Eigen::Vector3d p_cam;
      camSysPtr_->cam_left_ptr_->cam2World(coor, result[0], p_cam);
      dp.update_p_cam(p_cam);
      if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "l2") == 0 || strcmp(dpConfigPtr_->LSnorm_.c_str(), "zncc") == 0)
        dp.update(result[0], result[1]);
      else if(strcmp(dpConfigPtr_->LSnorm_.c_str(), "Tdist") == 0)
      {
        double scale2_rho = result[1] * (dpConfigPtr_->td_nu_ - 2) / dpConfigPtr_->td_nu_;
        dp.update_studentT(result[0], scale2_rho, result[1], dpConfigPtr_->td_nu_);
      }
      else
        exit(-1);
      dp.residual() = result[2];
      // dp.residual() = 1;
      dp.updatePose(T_world_virtual);
      job.vdpPtr_->push_back(dp);
    }
    return true;
  }


bool DepthProblemSolver::solve_single_problem_numerical(
  double d_init,
  std::shared_ptr< Eigen::NumericalDiff<DepthProblem> > & dProblemPtr,
  double* result)
{
  Eigen::VectorXd x(1);
  x << d_init;

  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<DepthProblem>, double> lm(*(dProblemPtr.get()));
  lm.resetParameters();
  lm.parameters.ftol = 1e-6;//1.E10*Eigen::NumTraits<double>::epsilon();
  lm.parameters.xtol = 1e-6;//1.E10*Eigen::NumTraits<double>::epsilon();
  lm.parameters.maxfev = dpConfigPtr_->MAX_ITERATION_ * 3;


  if(lm.minimizeInit(x) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters)
  {
    LOG(ERROR) << "ImproperInputParameters for LM (Mapping)." << std::endl;
    return false;
  }

  size_t iteration = 0;
  int optimizationState = 0;
  
  while(true)
  {
    Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeOneStep(x);
    iteration++;
      break;
    if(!slove_lr_)
    {
      // x(0) = d_init;
      // break;
    }
    bool terminate = false;
    if(status == 2 || status == 3)
    {
      switch (optimizationState)
      {
        case 0:
        {
          optimizationState++;
          break;
        }
        case 1:
        {
          terminate = true;
          break;
        }
      }
    }
    if(terminate)
      break;
  }

  // Since there is no way to set a optimization bound
  // on x in Eigen (as far as I know), a handy outlier rejection is applied here.
  if(x(0) <= 0.001)// we cannot see that far, right?
    return false;
  
  // update
  result[0] = x(0);
  // calculate the variance according to
  // https://android.googlesource.com/platform/external/eigen/+/jb-mr2-release/unsupported/test/NonLinearOptimization.cpp
  Eigen::internal::covar(lm.fjac, lm.permutation.indices());
  if(dpConfigPtr_->LSnorm_ == "l2" || dpConfigPtr_->LSnorm_ == "zncc")
  {
    double fnorm = lm.fvec.blueNorm();
    double covfac = fnorm * fnorm / (dProblemPtr->values() - dProblemPtr->inputs());
    Eigen::MatrixXd cov = covfac * lm.fjac.topLeftCorner<1,1>();
    result[1] = cov(0,0);
  }
  if(dpConfigPtr_->LSnorm_ == "Tdist")
  {
    Eigen::MatrixXd invSumJtT = lm.fjac.topLeftCorner<1,1>();
    result[1] = std::pow(dpConfigPtr_->td_stdvar_,2)  * lm.fjac(0, 0);//* invSumJtT(0,0);
  }
  result[2] = lm.fnorm * lm.fnorm;
  return true;
}

bool DepthProblemSolver::solve_single_problem_analytical(
  double d_init,
  std::shared_ptr< DepthProblem > & dProblemPtr,
  double* result)
{
  Eigen::VectorXd x(1);
  x << d_init;

  Eigen::LevenbergMarquardt<DepthProblem, double> lm(*(dProblemPtr.get()));
  lm.resetParameters();
  lm.parameters.ftol = 1e-6;//1.E10*Eigen::NumTraits<double>::epsilon();
  lm.parameters.xtol = 1e-6;//1.E10*Eigen::NumTraits<double>::epsilon();
  lm.parameters.maxfev = dpConfigPtr_->MAX_ITERATION_ * 3;

  if(lm.minimizeInit(x) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters)
  {
    LOG(ERROR) << "ImproperInputParameters for LM Analytical(Mapping)." << std::endl;
    return false;
  }

  size_t iteration = 0;
  int optimizationState = 0;
  double cost_begin = 0, cost_end = 0;
  while(true)
  {
    Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeOneStep(x);
    cost_begin = lm.fvec.sum();
    iteration++;
    if(iteration >= dpConfigPtr_->MAX_ITERATION_ )
      break;
    if(!slove_lr_)
    {
      // x(0) = d_init;
      break;
    }
    bool terminate = false;
    if(status == 2 || status == 3)
    {
      switch (optimizationState)
      {
        case 0:
        {
          optimizationState++;
          break;
        }
        case 1:
        {
          terminate = true;
          break;
        }
      }
    }
    if(terminate)
      break;
  }
  cost_end = lm.fvec.sum();
  // Since there is no way to set a optimization bound
  // on x in Eigen (as far as I know), a handy outlier rejection is applied here.
  if(x(0) <= 0.001)// we cannot see that far, right?
    return false;

  // update
  result[0] = x(0);
  Eigen::internal::covar(lm.fjac, lm.permutation.indices());
  if(dpConfigPtr_->LSnorm_ == "l2" || dpConfigPtr_->LSnorm_ == "zncc")
  {
    double fnorm = lm.fvec.blueNorm();
    double covfac = fnorm * fnorm / (dProblemPtr->values() - dProblemPtr->inputs());
    Eigen::MatrixXd cov = covfac * lm.fjac.topLeftCorner<1,1>();
    result[1] = cov(0,0);
  }
  if(dpConfigPtr_->LSnorm_ == "Tdist")
  {
    Eigen::MatrixXd invSumJtT = lm.fjac.topLeftCorner<1,1>();
    result[1] = std::pow(dpConfigPtr_->td_stdvar_,2) * invSumJtT(0,0);
  }
  result[2] = lm.fnorm * lm.fnorm;
  return true;
}

void
DepthProblemSolver::pointCulling(
  std::vector<DepthPoint> &vdp,
  double std_variance_threshold,
  double cost_threshold,
  double invDepth_min_range,
  double invDepth_max_range)
{
  std::vector<DepthPoint> vdp_culled;
  vdp_culled.reserve(vdp.size());
  std::vector<double> vDepth;
  vDepth.reserve(10000);
  for(size_t i = 0; i < vdp.size();i++)
  {
    if(vdp[i].variance() <= pow(std_variance_threshold,2) &&
      vdp[i].residual() <= cost_threshold &&
      vdp[i].valid() &&
      vdp[i].invDepth() >= invDepth_min_range &&
      vdp[i].invDepth() <= invDepth_max_range)
    {
      vdp_culled.push_back(vdp[i]);
      vDepth.emplace_back(1.0 / vdp[i].invDepth());
    }
  }
  vdp = vdp_culled;
#ifdef DEPTH_PROBLEM_SOLVER_LOG
  LOG(INFO) << "(culling) max depth: " << *std::max_element(vDepth.begin(), vDepth.end());
#endif
}

DepthProblemType DepthProblemSolver::getProblemType()
{
  return dpType_;
}

}//end of namespace core
}//end of namespace esvo2_core

