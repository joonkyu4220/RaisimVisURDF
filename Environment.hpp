#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {
  public:
    explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
        RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
      setup(cfg);
      setWorld();
      setCharacter();
      setData();
      setAgent(cfg);
    }

  void init() final {}

  void setup(const Yaml::Node& cfg){
    // EXPERIMENT SETTINGS
    charFileName_ = cfg["character"]["file name"].template As<std::string>();
    motionFileName_ = cfg["motion data"]["file name"].template As<std::string>();
    dataHasWrist_ = cfg["motion data"]["has wrist"].template As<bool>();
    control_dt_ = 1.0 / cfg["motion data"]["fps"].template As<float>();
    simulation_dt_ = cfg["simulation_dt"].template As<float>();
  }

  void setWorld(){
    world_ = std::make_unique<raisim::World>();
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
    }
    world_->addGround(0, "steel");
    world_->setERP(1.0);
    world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);
  }

  void setCharacter(){
    // CHARACTER SETUP
    simChar_ = world_->addArticulatedSystem(resourceDir_ + "/" + charFileName_ + ".urdf"); 
    simChar_->setName("sim character");
    if (visualizable_){
      server_->focusOn(simChar_);
    }

    gcDim_ = simChar_->getGeneralizedCoordinateDim(); // 51
    gc_.setZero(gcDim_); gcInit_.setZero(gcDim_); gcRef_.setZero(gcDim_);

    gvDim_ = simChar_->getDOF(); // 40
    controlDim_ = gvDim_ - 6;
    gv_.setZero(gvDim_); gvInit_.setZero(gvDim_); gvRef_.setZero(gvDim_);
    
    com_.setZero(comDim_); comRef_.setZero(comDim_);
    ee_.setZero(eeDim_); eeRef_.setZero(eeDim_);
    
    for (auto bodyName:simChar_->getBodyNames()){
      std::cout << bodyName << std::endl;
    }
  }

  void setData(){
    // DATA PREPARATION
    loadData();
    if (isPreprocess_) {
      preprocess();
    }
    loadGT();
  }

  void setAgent(const Yaml::Node& cfg){
    obDim_ = 1;
    obDouble_.setZero(obDim_);
    stateDim_ = 1;
    stateDouble_.setZero(stateDim_);
    actionDim_ = 1;
    rewards_.initializeFromConfigurationFile(cfg_["reward"]);
  }

  void loadData(){
    dataGC_.setZero(maxLen_, gcDim_);
    std::ifstream gcfile(resourceDir_ + "/" + motionFileName_ + ".txt");
    float data;
    int row = 0, col = 0;
    while (gcfile >> data) {
      dataGC_.coeffRef(row, col) = data;
      col++;
      if (!dataHasWrist_ && (col == cStart_[rWristIdx_] || col == cStart_[lWristIdx_])){ // skip the wrist joints
        dataGC_.coeffRef(row, col) = 1; dataGC_.coeffRef(row, col + 1) = 0; dataGC_.coeffRef(row, col + 2) = 0; dataGC_.coeffRef(row, col + 3) = 0;
        col += 4;
      }
      if (col == gcDim_){
        col = 0;
        row++;
      }
    }
    dataLen_ = row;
    dataGC_ = dataGC_.topRows(dataLen_);
  }

  void preprocess(){
    dataGV_.setZero(dataLen_, gvDim_);
    dataEE_.setZero(dataLen_, eeDim_);
    dataCom_.setZero(dataLen_, comDim_);

    Mat<3, 3> rootRotInv;
    Vec<3> rootPos, jointPos_W, jointPos_B, comPos_W, comPos_B;
    
    // SOLVE FK FOR EE & COM
    for(int frameIdx = 0; frameIdx < dataLen_; frameIdx++) {
      simChar_->setState(dataGC_.row(frameIdx), gvInit_);
      simChar_->getState(gc_, gv_);
      getRootTransform(rootRotInv, rootPos);
      
      int eeIdx = 0;
      for (int bodyIdx = 1; bodyIdx < nJoints_ + 1; bodyIdx ++){
        if (isEE_[bodyIdx - 1]){
          simChar_->getBodyPosition(bodyIdx, jointPos_W);
          matvecmul(rootRotInv, jointPos_W - rootPos, jointPos_B);
          dataEE_.row(frameIdx).segment(eeIdx, 3) = jointPos_B.e();
          eeIdx += 3;
        }
      }
      // center-of-mass (world-frame!)
      comPos_W = simChar_->getCOM();
      dataCom_.row(frameIdx).segment(0, 3) = comPos_W.e();
    }
    
    // CALCULATE ANGULAR VELOCITY FOR GV
    Eigen::VectorXd prevFrame, nextFrame, prevGC, nextGC;
    for (int frameIdx = 0; frameIdx < dataLen_; frameIdx++){
      int prevFrameIdx = std::max(frameIdx - 1, 0);
      int nextFrameIdx = std::min(frameIdx + 1, dataLen_ - 1);
      Eigen::VectorXd prevFrame = dataGC_.row(prevFrameIdx), nextFrame = dataGC_.row(nextFrameIdx);
      float dt = (nextFrameIdx - prevFrameIdx) * control_dt_;

      // root position
      prevGC = prevFrame.segment(0, 3); nextGC = nextFrame.segment(0, 3);
      dataGV_.row(frameIdx).segment(0, 3) = (nextGC - prevGC) / dt;
      // root orientation
      prevGC = prevFrame.segment(3, 4); nextGC = nextFrame.segment(3, 4);
      dataGV_.row(frameIdx).segment(3, 3) = getAngularVelocity(prevGC, nextGC, dt);

      for (int jointIdx = 0; jointIdx < nJoints_; jointIdx++){
        prevGC = prevFrame.segment(cStart_[jointIdx], cDim_[jointIdx]);
        nextGC = nextFrame.segment(cStart_[jointIdx], cDim_[jointIdx]);
        if (cDim_[jointIdx] == 1) {
          dataGV_.row(frameIdx).segment(vStart_[jointIdx], vDim_[jointIdx]) = (nextGC - prevGC) / dt;
        }
        else {
          dataGV_.row(frameIdx).segment(vStart_[jointIdx], vDim_[jointIdx]) = getAngularVelocity(prevGC, nextGC, dt);
        }
      }
    }

    // WRITE FILES
    std::ofstream gvFile(resourceDir_ + "/" + motionFileName_ + "_gv.txt", std::ios::out | std::ios::trunc);
    if (gvFile){
      gvFile << dataGV_;
      gvFile.close();
    }
    std::ofstream eeFile(resourceDir_ + "/" + motionFileName_ + "_ee.txt", std::ios::out | std::ios::trunc);
    if (eeFile){
      eeFile << dataEE_;
      eeFile.close();
    }
    std::ofstream comFile(resourceDir_ + "/" + motionFileName_ + "_com.txt", std::ios::out | std::ios::trunc);
    if (comFile){
      comFile << dataCom_;
      comFile.close();
    }
  }

  Eigen::VectorXd getAngularVelocity(Eigen::VectorXd prev, Eigen::VectorXd next, float dt){
    Eigen::Quaterniond prevq, nextq, diffq;
    prevq.w() = prev(0); prevq.vec() = prev.segment(1, 3);
    nextq.w() = next(0); nextq.vec() = next.segment(1, 3);
    diffq = prevq.inverse() * nextq;
    float deg, gain;
    if (std::abs(diffq.w()) > 0.999999f) {
      gain = 0;
    }
    else if (diffq.w() < 0){
      deg = std::acos(-diffq.w());
      gain = -2.0f * deg / (std::sin(deg) * dt);
    }
    else{
      deg = std::acos(diffq.w());
      gain = 2.0f * deg / (std::sin(deg) * dt);
    }
    return gain * diffq.vec();
  }

  void loadGT(){
    dataGV_.setZero(dataLen_, gvDim_);
    std::ifstream gvfile(resourceDir_ + "/" + motionFileName_ + "_gv.txt");
    float data;
    int row = 0, col = 0;
    while (gvfile >> data) {
      dataGV_.coeffRef(row, col) = data;
      col++;
      if (col == gvDim_){
        col = 0;
        row++;
      }
    }

    dataEE_.setZero(dataLen_, eeDim_);
    std::ifstream eefile(resourceDir_ + "/" + motionFileName_ + "_ee.txt");
    row = 0, col = 0;
    while (eefile >> data) {
      dataEE_.coeffRef(row, col) = data;
      col++;
      if (col == eeDim_){
        col = 0;
        row++;
      }
    }

    dataCom_.setZero(dataLen_, comDim_);
    std::ifstream comfile(resourceDir_ + "/" + motionFileName_ + "_com.txt");
    row = 0, col = 0;
    while (comfile >> data) {
      dataCom_.coeffRef(row, col) = data;
      col++;
      if (col == comDim_){
        col = 0;
        row++;
      }
    }

    // TODO: integrate loop transformation code
    std::ifstream loopdispfile(resourceDir_ + "/" + motionFileName_ + "_loop_disp.txt");
    row = 0, col = 0;
    loopDisplacement_.setZero();
    while (loopdispfile >> data) {
      loopDisplacement_[col] = data;
      col++;
    }

    std::ifstream loopturnfile(resourceDir_ + "/" + motionFileName_ + "_loop_turn.txt");
    Vec<4> loop_turn_quat;
    loop_turn_quat.setZero();
    loop_turn_quat[0] = 1;
    row = 0, col = 0;
    while (loopturnfile >> data) {
      loop_turn_quat[col] = data;
      col++;
    }
    raisim::quatToRotMat(loop_turn_quat, loopTurn_);
  }

  void reset() final {
    sim_step_ = 0;
    total_reward_ = 0;
    
    initializeCharacter();

    updateObservation();
  }

  void initializeCharacter(){
    // index_ = rand() % dataLen_;
    index_ = 0;
    gcInit_ = dataGC_.row(index_);
    gvInit_ = dataGV_.row(index_);
    
    simChar_->setState(gcInit_, gvInit_);
  }

  void updateObservation() {
    
  }

  void getRootTransform(Mat<3,3>& rot, Vec<3>& pos) {
    Vec<4> rootRot, defaultRot, rootRotRel;
    rootRot[0] = gc_[3]; rootRot[1] = gc_[4]; rootRot[2] = gc_[5]; rootRot[3] = gc_[6];
    defaultRot[0] = 0.707; defaultRot[1] =  - 0.707; defaultRot[2] = 0; defaultRot[3] = 0;
    raisim::quatMul(defaultRot, rootRot, rootRotRel);
    double yaw = atan2(2 * (rootRotRel[0] * rootRotRel[2] + rootRotRel[1] * rootRotRel[3]), 1 - 2 * (rootRotRel[2] * rootRotRel[2] + rootRotRel[3] * rootRotRel[3]));
    Vec<4> quat;
    quat[0] = cos(yaw / 2); quat[1] = 0; quat[2] = 0; quat[3] = - sin(yaw / 2);
    raisim::quatToRotMat(quat, rot);
    pos[0] = gc_[0]; pos[1] = gc_[1]; pos[2] = gc_[2];
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    gcRef_ = dataGC_.row(index_);
    gvRef_ = dataGV_.row(index_);
    simChar_->setState(gcRef_, gvRef_);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
    {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    updateObservation();
    updateTargetMotion();
    return 0;
  }

  void updateTargetMotion(){
    index_ += 1;
    sim_step_ += 1;
    if (index_ >= dataLen_){
      index_ = 0;
    }
  }

  void computeReward() {
  }
  
  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }

  void getState(Eigen::Ref<EigenVec> state) final {
    state = stateDouble_.cast<float>();
  }

  bool time_limit_reached() {
    return sim_step_ > max_sim_step_;
  }

  float get_total_reward() {
    return float(total_reward_);
  }

  bool isTerminalState(float& terminalReward) final {
    return false;
  }

  private:
    std::string motionFileName_, charFileName_;
    bool dataHasWrist_, isPreprocess_;
    bool visualizable_ = true;
    raisim::ArticulatedSystem *simChar_;

    int nJoints_ = 14;
    
    int chestIdx_ = 0;
    int neckIdx_ = 1;
    int rShoulderIdx_ = 2;
    int rElbowIdx_ = 3;
    int rWristIdx_ = 4;
    int lShoulderIdx_ = 5;
    int lElbowIdx_ = 6;
    int lWristIdx_ = 7;
    int rHipIdx_ = 8;
    int rKneeIdx_ = 9;
    int rAnkleIdx_ = 10;
    int lHipIdx_ = 11;
    int lKneeIdx_ = 12;
    int lAnkleIdx_ = 13;

    int cStart_[14] = {7, 11,  15, 19, 20,  24, 28, 29,  33, 37, 38,  42, 46, 47};
    int vStart_[14] = {6,  9,  12, 15, 16,  19, 22, 23,  26, 29, 30,  33, 36, 37};
    int cDim_[14] = {4, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4,  4, 1, 4};
    int vDim_[14] = {3, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3,  3, 1, 3};
    int isEE_[14] = {0, 0,  0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1};
    int isRightArm_[14] = {0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  0, 0, 0};

    Vec<3> rHandCenter = {0, -0.08850, 0};

    int gcDim_, gvDim_, controlDim_;
    int posDim_ = 3 * nJoints_, comDim_ = 3, eeDim_ = 12;

    Eigen::VectorXd gc_, gv_, gcInit_, gvInit_, gcRef_, gvRef_;
    Eigen::MatrixXd dataGC_, dataGV_, dataEE_, dataCom_;
    Eigen::VectorXd com_, comRef_, ee_, eeRef_;
    
    Vec<3> loopDisplacement_;
    Mat<3,3> loopTurn_;
    Vec<3> loopDisplacementAccumulated_;
    Mat<3,3> loopTurnAccumulated_;

    Eigen::VectorXd pTarget_, vTarget_;
    Eigen::VectorXd obDouble_, stateDouble_;

    int maxLen_ = 1000;
    int dataLen_;
    int index_ = 0;

    int sim_step_ = 0;
    int max_sim_step_ = 2000;
    double total_reward_ = 0;
};

}