//
// Created by tipakorng on 3/3/16.
//Modified by Lexi Scott in 2022
//

#include <visualization_msgs/MarkerArray.h>
#include "simulator.h"
#include "active_sensing_continuous_local/ObsrvBack.h"
#include "active_sensing_continuous_local/ReqObsrv.h"
#include "active_sensing_continuous_local/UpdateInfo.h"
#include "active_sensing_continuous_local/CT.h"
#include "active_sensing_continuous_local/action_message.h"
#include "rng.h"
#include<iostream>
#include<thread>
using namespace std;
unsigned int sensing_action_global;
double observation_global;
int req_obversation_flag = 0;
int update_flag = 0;
double update_location_x;
double update_location_y;
double update_location_z;
int communication_count = 0;
int sensing_action_local;
int ncount = 0;
int timeout =0;
bool paused = false;
unsigned int sensing_action;
Eigen::VectorXd observation(1);
Eigen::VectorXd task_action(3);
int Break_Point = 0;
int Observation_point = 0;
int totalClients = 0;
bool recovery = false;
Simulator::Simulator(Model &model, BeliefSpacePlanner &planner, unsigned int sensing_interval) :
        model_(model),
        planner_(planner),
        sensing_interval_(sensing_interval)
{
    has_publisher_ = false;
}

Simulator::Simulator(Model &model, BeliefSpacePlanner &planner, ros::NodeHandle *node_handle,
                     unsigned int sensing_interval) :
        model_(model),
        planner_(planner),
        sensing_interval_(sensing_interval),
        node_handle_(node_handle)
{
    publisher_ = node_handle_->advertise<visualization_msgs::Marker>("simulator", 1);
    has_publisher_ = true;
}

Simulator::~Simulator()
{}

void Simulator::initSimulator()
{
    // Clear stuff
    states_.clear();
    task_actions_.clear();
    sensing_actions_.clear();
    observations_.clear();
    cumulative_reward_ = 0;
    // Reset planner
    planner_.reset();
    // Reset active sensing time
    active_sensing_time_ = 0;
}

void Simulator::updateSimulator(unsigned int sensing_action, Eigen::VectorXd observation, Eigen::VectorXd task_action)
{
	globalcheckflag = 1;
	nextflag=1;
        ROS_INFO("update simulator");
    Eigen::VectorXd new_state = model_.sampleNextState(states_.back(), task_action);
    ROS_INFO("new state0 is %f",new_state(0));
    ROS_INFO("new state1 is %f",new_state(1));
	nextflag=0;
    states_.push_back(new_state);
    sensing_actions_.push_back(sensing_action);
    observations_.push_back(observation);
    task_actions_.push_back(task_action);
    cumulative_reward_ += model_.getReward(new_state, task_action);
	globalcheckflag = 0;
}

void Simulator::updateSimulator(Eigen::VectorXd task_action)
{
	globalcheckflag = 1;
	nextflag=1;
    ROS_INFO("updateelse simulator");
    Eigen::VectorXd new_state = model_.sampleNextState(states_.back(), task_action);
    ROS_INFO("new state0 is %f",new_state(0));
    ROS_INFO("new state1 is %f",new_state(1));
	nextflag=0;
    states_.push_back(new_state);
    task_actions_.push_back(task_action);
    cumulative_reward_ += model_.getReward(new_state, task_action);
	globalcheckflag = 0;
}


std::chrono::high_resolution_clock::time_point taskaction_start;
std::chrono::high_resolution_clock::time_point taskaction_finish;
std::chrono::duration<double> taskaction_elapsed_time;

std::chrono::high_resolution_clock::time_point active_sensing_start;
std::chrono::high_resolution_clock::time_point active_sensing_finish;
std::chrono::duration<double> active_sensing_elapsed_time;



bool Simulator::localmachine(active_sensing_continuous_local::action_message::Request &req,
                  active_sensing_continuous_local::action_message::Response &res)
{
    totalClients +=1;
    paused = true;
    timeout = 0;
    ROS_INFO("SET TIME OUT EQUAL TO ZERO");
    ROS_INFO("IN PROCESSING.");
    ROS_INFO("INCREASED TOTAL CLIENTS %d", totalClients);
  
    if(req.source == 1 & recovery){
        ROS_INFO("GOT SERVER WHILE RUNNING LOCAL.");
        ros::Duration(2).sleep();
    }
    switch(req.type)
  {
    case 1:
        //planner_.normalizeBelief();
        //sensing_action = planner_.getSensingAction();
        ROS_INFO("RECEIVE SENSING ACTION.");
        ROS_INFO("GETTING OBSERVATION.");
        //res.x1 = model_.sampleObservation(states_.back(), req.x);
        sensing_action_local=req.x;
        active_sensing_finish = std::chrono::high_resolution_clock::now();
        if(Observation_point==0)
        {
        observation = model_.sampleObservation(states_.back(), sensing_action_local);
        planner_.updateBelief(sensing_action, observation);
        task_action = planner_.getTaskAction(); 
        Observation_point=1;
        }
        res.x1 = observation(0);
        res.type1 = 2;
        //observation = model_.sampleObservation(states_.back(), sensing_action);
        //planner_.updateBelief(sensing_action, observation);
	    //task_action = planner_.getTaskAction(); 
        taskaction_start = std::chrono::high_resolution_clock::now();
        paused = false;
        break;
    case 3:
        ROS_INFO("RECEIVE TASK ACTION.");
        ROS_INFO("PERFORMING");
        ROS_INFO("TELL AWS CONTINUE");
        task_action(0) = req.x;
        task_action(1) = req.y;
        task_action(2) = req.z;
        taskaction_finish = std::chrono::high_resolution_clock::now();
        res.x1 = 0;
        res.y1 = 0;
        res.z1 = 0;
        res.type1 = 4;
        planner_.predictBelief(task_action);
        updateSimulator(sensing_action, observation, task_action);
        ncount++;
        communication_count++;
        Break_Point=1;
        paused = false;
        break;
  }

}

void timer(){
    while(!paused){
        ROS_INFO("WE COUNTING %d", timeout);
        ros::Duration(2).sleep();
        timeout += 1;
        if(timeout >= 10){
            ROS_ERROR("LOST CONNECTION TO SERVER SWITCH TO LOCAL RECOVERY");
            
            ofstream fw("recoveryclient.txt", std::ofstream::out);
            if (fw.is_open()){
                fw << endl;
                //fw << ("Communcation count is ");
                fw << (communication_count) << endl;
                //fw << ("N count is");
                fw<< (ncount) << endl;
                // fw << ("Sensing action local "); 
                fw << sensing_action_local << endl;
                //fw << ("Sensing action global ");
                fw << sensing_action_global << endl;
                //fw << ("Sensing action ");
                fw << sensing_action << endl;
            }
                fw.close();
            system("./serverRecovery.sh");
            exit(0);

        }

    }

}

void Simulator::simulate(const Eigen::VectorXd &init_state, unsigned int num_steps, unsigned int verbosity, bool recovery1)
{
    initSimulator();
    ncount=0;

    double active_sensing_time = 0;
    double observation_time = 0;
    double updatebelief_time =0;
    double taskaction_time = 0;
    double predictbelief_time = 0;
    double total_updatebelief_time = 0;
    double total_predictbelief_time = 0;

    double avg_observation_time = 0;
    double avg_updatebelief_time = 0;
    double avg_total_updatebelief_time=0;
    double avg_taskaction_time = 0;
    double avg_predictbelief_time = 0;
    double avg_total_predictbelief_time = 0;
    recovery = recovery1;
    std::cout<< "RECOVERY VARIABLE IS";
    std::cout<<recovery;
    if(recovery){
        std::cout<<"Entering from recovery mode, using variables -";
        ifstream inFile("recoveryclient.txt", ios::in);
       
        inFile >> communication_count, ncount, sensing_action_local, sensing_action_global, sensing_action;
        std::cout <<"N COUNT SHOULD BE PRINTING JUST BELOW THIS LOOK LOOK";
        std::cout<<ncount;


        /*ncount = 10;
        communication_count = 10;
        sensing_action_global = 0;
        sensing_action_local = 0;
        sensing_action = 1;
        */
    }

    states_.push_back(init_state);

    std::chrono::high_resolution_clock::time_point observation_start;
    std::chrono::high_resolution_clock::time_point observation_finish;
    std::chrono::high_resolution_clock::time_point updatebelief_start;
    std::chrono::high_resolution_clock::time_point updatebelief_finish;

    std::chrono::high_resolution_clock::time_point predictbelief_start;
    std::chrono::high_resolution_clock::time_point predictbelief_finish;
    std::chrono::duration<double> observation_elapsed_time;
    std::chrono::duration<double> updatebelief_elapsed_time;
    std::chrono::duration<double> predictbelief_elapsed_time;
    std::chrono::duration<double> total_updatebelief_elapsed_time;
    std::chrono::duration<double> total_predictbelief_elapsed_time;
    if (verbosity > 0)
        std::cout << "state = \n" << states_.back().transpose() << std::endl;

    planner_.publishParticles();


    ros::NodeHandle nh;
    /*ros::ServiceClient client = n.serviceClient<active_sensing_continues_local::action_message>("add_two_ints");
    active_sensing_continues_local::action_message srv;*/
    
    ros::ServiceServer service = nh.advertiseService("new_active_sensing",&Simulator::localmachine,this);
    ROS_INFO("Advertised service");
    std::thread thread_obj(timer);
    thread_obj.detach();
    while(ros::ok&&!model_.isTerminal(states_.back()) && ncount < num_steps)
    {
        planner_.normalizeBelief();
        ROS_INFO("communication round is %d",communication_count);
	    ROS_INFO("n is %d",ncount);
        Observation_point=0;
        //Check for the roscore
         if(!ros::master::check()){
            ROS_ERROR("Failed to access roscore on round %d", communication_count);
            ros::Duration(2).sleep();
              ofstream fw("recoveryclient.txt", std::ofstream::out);
            if (fw.is_open()){
                fw << endl;
                //fw << ("Communcation count is ");
                fw << (communication_count) << endl;
                //fw << ("N count is");
                fw<< (ncount) << endl;
                // fw << ("Sensing action local "); 
                fw << sensing_action_local << endl;
                //fw << ("Sensing action global ");
                fw << sensing_action_global << endl;
                //fw << ("Sensing action ");
                fw << sensing_action << endl;
            }
                fw.close();
            system("sudo ./rosrecovery.sh");
            exit(0);
        }
        if (ncount % (sensing_interval_ + 1) == 0 && ros::master::check())
        {
            //planner_.normalizeBelief();
	        //ROS_INFO("communication round is %d",communication_count);
	        //ROS_INFO("n is %d",ncount);
            sensing_action = planner_.getSensingAction();
            active_sensing_start = std::chrono::high_resolution_clock::now();
            while(ros::ok)
            {
            Break_Point = 0;
            ros::spinOnce();
            //ros::Duration(1.0).sleep();
            if(Break_Point==1)
            {
                break;
            }
            }
            taskaction_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >
            (taskaction_finish-taskaction_start);
            taskaction_time += taskaction_elapsed_time.count();

            active_sensing_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >
                    (active_sensing_finish-active_sensing_start);
            active_sensing_time += active_sensing_elapsed_time.count();
            //observation = model_.sampleObservation(states_.back(), sensing_action);
            //planner_.updateBelief(sensing_action, observation);
	        //task_action = planner_.getTaskAction();
                if (verbosity > 0)
                {
                    std::cout << "n = " << ncount << std::endl;
                    std::cout << "sensing_action = " << sensing_action << std::endl;
                    std::cout << "observation = " << observation.transpose() << std::endl;
                    std::cout << "most_likely_state = " << planner_.getMaximumLikelihoodState().transpose() << std::endl;
                    std::cout << "task_action = " << task_action.transpose() << std::endl;
                    std::cout << "state = " << states_.back().transpose() << std::endl;
                }

        }
        else
        {
                //planner_.normalizeBelief();
                ROS_INFO("communication round is %d",communication_count);
	            ROS_INFO("n is %d",ncount);
                task_action = planner_.getTaskAction();
                ROS_INFO("task_action(0) is %f",task_action(0));
                ROS_INFO("task_action(1) is %f",task_action(1));
                planner_.predictBelief(task_action);
                updateSimulator(task_action);
                if (verbosity > 0)
                {
                    std::cout << "n = " << ncount << std::endl;
                    std::cout << "sensing_action = " << sensing_action << std::endl;
                    std::cout << "observation = " << observation.transpose() << std::endl;
                    std::cout << "most_likely_state = " << planner_.getMaximumLikelihoodState().transpose() << std::endl;
                    std::cout << "task_action = " << task_action.transpose() << std::endl;
                    std::cout << "state = " << states_.back().transpose() << std::endl;
                }
                ncount++;
                communication_count++;
        }
            planner_.publishParticles();
            model_.publishMap();
            if (has_publisher_)
            {
                publishState();
            }
	        ROS_INFO("!model_.isTerminal(states_.back()) after loop is %d",!model_.isTerminal(states_.back()));
	        ROS_INFO("\n");
            
        //ros::Duration(1.0).sleep();
    }
    int num_sensing_steps = (ncount + 1) / (sensing_interval_ + 1);

    if (num_sensing_steps > 0)
    {
        active_sensing_time_ = active_sensing_time / num_sensing_steps;
        taskaction_time_ = taskaction_time / num_sensing_steps;
    }
    else
    {
        active_sensing_time_ = 0;
        taskaction_time_ = 0;
    }
   
}



std::vector<Eigen::VectorXd> Simulator::getStates()
{
    return states_;
}

std::vector<unsigned int> Simulator::getSensingActions()
{
    return sensing_actions_;
}

std::vector<Eigen::VectorXd> Simulator::getTaskActions()
{
    return task_actions_;
}

std::vector<Eigen::VectorXd> Simulator::getObservations()
{
    return observations_;
}

double Simulator::getCumulativeReward()
{
    return cumulative_reward_;
}

double Simulator::getAverageActiveSensingTime()
{
    return active_sensing_time_;
}


double Simulator::getAvgTaskactionTime()
{
return taskaction_time_;
}

void Simulator::publishState()
{
    if (has_publisher_)
    {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(0, 0, 0));
        tf::Quaternion q;
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "my_frame"));

        visualization_msgs::Marker marker;

        uint32_t shape = visualization_msgs::Marker::CUBE;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.ns = "basic_shapes";
        marker.id = 0;
        marker.type = shape;
        marker.action = visualization_msgs::Marker::ADD;

        Eigen::VectorXd state = states_.back();

        model_.fillMarker(state, marker);

        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        marker.lifetime = ros::Duration();

        publisher_.publish(marker);
    }
}
