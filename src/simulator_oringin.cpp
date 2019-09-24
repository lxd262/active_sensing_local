//
// Created by tipakorng on 3/3/16.
//

#include <visualization_msgs/MarkerArray.h>
#include "simulator.h"
#include "active_sensing_continuous_local/ObsrvBack.h"
#include "active_sensing_continuous_local/ReqObsrv.h"
#include "active_sensing_continuous_local/UpdateInfo.h"

unsigned int sensing_action_global;
double observation_global;
int req_obversation_flag = 0;
int update_flag = 0;
double update_location_x;
double update_location_y;
double update_location_z;

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
    Eigen::VectorXd new_state = model_.sampleNextState(states_.back(), task_action);
    states_.push_back(new_state);
    sensing_actions_.push_back(sensing_action);
    observations_.push_back(observation);
    task_actions_.push_back(task_action);
    cumulative_reward_ += model_.getReward(new_state, task_action);
}

void Simulator::updateSimulator(Eigen::VectorXd task_action)
{
    Eigen::VectorXd new_state = model_.sampleNextState(states_.back(), task_action);
    states_.push_back(new_state);
    task_actions_.push_back(task_action);
    cumulative_reward_ += model_.getReward(new_state, task_action);
}

void reqObservationCallback(const active_sensing_continuous_local::ReqObsrv& msg)
{
  ROS_INFO("local heard");
  ROS_INFO("observation: [%d]", msg.observe_there);
  sensing_action_global = msg.observe_there;
  req_obversation_flag = 1;
  ROS_INFO("flag is %d", req_obversation_flag);
  ROS_INFO("local finish \n");
}

void updateCallback(const active_sensing_continuous_local::UpdateInfo& msg)
{
  ROS_INFO("task action recieved");
  update_location_x = msg.x;
  update_location_y = msg.y;
  update_location_z = msg.z;
  update_flag = 1;
  ROS_INFO("flag is %d", update_flag);
}

void Simulator::simulate(const Eigen::VectorXd &init_state, unsigned int num_steps, unsigned int verbosity)
{
    initSimulator();
    unsigned int n = 0;
    unsigned int sensing_action;
    double active_sensing_time = 0;
    Eigen::VectorXd observation(1);
    Eigen::VectorXd task_action(3);
    states_.push_back(init_state);
    std::chrono::high_resolution_clock::time_point active_sensing_start;
    std::chrono::high_resolution_clock::time_point active_sensing_finish;
    std::chrono::duration<double> active_sensing_elapsed_time;

    if (verbosity > 0)
        std::cout << "state = \n" << states_.back().transpose() << std::endl;

    planner_.publishParticles();

    while (!model_.isTerminal(states_.back()) && n < num_steps)
    {
        // Normalize particle weights.
        planner_.normalizeBelief();

        // If sensing is allowed in this step.
        if (n % (sensing_interval_ + 1) == 0)
        {
            /*
                code below use spin loops to set timer that enable synchronization
            */

            ROS_INFO("REST FOR A WHILE!!!!!!!!");
            int counter_0 = 0;
            while(counter_0 < 100){
                counter_0++;
                ros::spinOnce();
            }
            ROS_INFO("REST DONE");

            /*
                code above use spin loops to set timer that enable synchronization
            */
            
            // Get sensing action.
            active_sensing_start = std::chrono::high_resolution_clock::now();
            //sensing_action = planner_.getSensingAction();

            /*
                code below get sensing action    
            */

            ros::NodeHandle n_h;
	    
            ros::Subscriber sub = n_h.subscribe("req_obsrv", 1000, reqObservationCallback);
            ros::Rate loop_rate(10);
            ROS_INFO("create a subber");

            while(true){
                ros::spinOnce();
		ROS_INFO("subber1 spin once");
                loop_rate.sleep();
                if(req_obversation_flag == 1){
                    break;
                }
            }
            ROS_INFO("get out of subber loop");

            req_obversation_flag = 0;
            ROS_INFO("req_obversation_flag is %d", req_obversation_flag);
            sensing_action = sensing_action_global;

            /*
                code above get sensing action
            */

            active_sensing_finish = std::chrono::high_resolution_clock::now();
            active_sensing_elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >
                    (active_sensing_finish-active_sensing_start);
            active_sensing_time += active_sensing_elapsed_time.count();
           
            // Update the belief.
            observation = model_.sampleObservation(states_.back(), sensing_action);

            /*
                code below use spin loops to set timer that enable synchronization
            */

            ROS_INFO("REST FOR A WHILE!!!!!!!!");
            int counter_a = 0;
            while(counter_a < 100){
                counter_a++;
                ros::spinOnce();
            }
            ROS_INFO("REST DONE");

            /*
                code above use spin loops to set timer that enable synchronization
            */


            /*
                code below publish observation
            */
            
            observation_global = observation(0);
            ros::Publisher observation_pub = n_h.advertise<active_sensing_continuous_local::ObsrvBack>("obversation_back", 1000);
//ros::Duration(3.0).sleep();
	   ROS_INFO("publish obversation_back");
            ROS_INFO("get into while(ros::ok()) loop");
            while (ros::ok())
            {
                int connections = observation_pub.getNumSubscribers();
                ROS_INFO("connected: %d", connections);
                active_sensing_continuous_local::ObsrvBack msg;
                msg.observation_back = observation_global;
                ROS_INFO("observe back: %f", observation_global);
                if(connections > 0){
                    int i = 0;
                    while(i < 10){
                        observation_pub.publish(msg);
                        i++;
                        ros::spinOnce();
                    }
                ROS_INFO("published");
                break;
                }
                loop_rate.sleep();
            }
            ROS_INFO("get out of while(ros::ok()) loop");

            /*
                code above publish observation
            */

            //Eigen::IOFormat CommaInitFmt(4, 0, ", ", ", ", "", "", " << ", ";");
            //std::cout << "observation is " << observation.format(CommaInitFmt) << std::endl;

            //planner_.updateBelief(sensing_action, observation);

            // Predict the new belief.
            //task_action = planner_.getTaskAction();


            /*
                code below use spin loops to set timer that enable synchronization
            */

            ROS_INFO("REST FOR A WHILE!!!!!!!!");
            int counter_b = 0;
            while(counter_b < 100){
                counter_b++;
                ros::spinOnce();
            }
            ROS_INFO("REST DONE");

            /*
                code above use spin loops to set timer that enable synchronization
            */


            /*
                code below get tast action
            */
	   ros::Duration(3.0).sleep();
            ros::Subscriber sub3 = n_h.subscribe("update", 1000, updateCallback);
            ROS_INFO("create a subber3");

            while(true){
                ros::spinOnce();
                loop_rate.sleep();
                if(update_flag == 1){
                    break;
                }
            }
            ROS_INFO("get out of subber3 loop");

            update_flag = 0;
            ROS_INFO("update_flag is %d", update_flag);
            task_action(0) = update_location_x;
            task_action(1) = update_location_y;
            task_action(2) = update_location_z;

            /*
                code above get tast action
            */

            planner_.predictBelief(task_action);

            //std::cout << "task_action is " << task_action.format(CommaInitFmt) << std::endl;

            updateSimulator(sensing_action, observation, task_action);

            if (verbosity > 0)
            {
                std::cout << "n = " << n << std::endl;
                std::cout << "sensing_action = " << sensing_action << std::endl;
                std::cout << "observation = " << observation.transpose() << std::endl;
                std::cout << "most_likely_state = " << planner_.getMaximumLikelihoodState().transpose() << std::endl;
                std::cout << "task_action = " << task_action.transpose() << std::endl;
                std::cout << "state = " << states_.back().transpose() << std::endl;
            }
        }

        // Otherwise, set sensing action to DO NOTHING.
        else
        {
            //task_action = planner_.getTaskAction();
                        
            /*
                code below use spin loops to set timer that enable synchronization
            */
            ros::NodeHandle n_h;
            ros::Rate loop_rate(10);

            ROS_INFO("REST FOR A WHILE!!!!!!!!");
            int counter_b = 0;
            while(counter_b < 100){
                counter_b++;
                ros::spinOnce();
            }
            ROS_INFO("REST DONE");

            /*
                code above use spin loops to set timer that enable synchronization
            */


            /*
                code below get tast action
            */
	    ros::Duration(3.0).sleep();
            ros::Subscriber sub4 = n_h.subscribe("updateelse", 1000, updateCallback);
            ROS_INFO("create a subber4");
	    //ros::AsyncSpinner spinner(4); 
	    int counterdebug=0;
            while(true){
                ros::spinOnce();
		//spinner.start();
                loop_rate.sleep();
		//ros::waitForShutdown();
		counterdebug++;
		ROS_INFO("check counts is %d",counterdebug);
                if(update_flag == 1){
                    break;
                }
            }
            ROS_INFO("get out of subber4 loop");

            update_flag = 0;
            ROS_INFO("update_flag is %d", update_flag);
            task_action(0) = update_location_x;
            task_action(1) = update_location_y;
            task_action(2) = update_location_z;

            /*
                code above get tast action
            */

            planner_.predictBelief(task_action);
            updateSimulator(task_action);

            if (verbosity > 0)
            {
                std::cout << "n = " << n << std::endl;
                std::cout << "sensing_action = " << "n/a" << std::endl;
                std::cout << "observation = " << "n/a" << std::endl;
                std::cout << "most_likely_state = " << planner_.getMaximumLikelihoodState().transpose() << std::endl;
                std::cout << "task_action = " << task_action.transpose() << std::endl;
                std::cout << "state = " << states_.back().transpose() << std::endl;
            }
        }

        planner_.publishParticles();
        model_.publishMap();

        if (has_publisher_)
        {
            publishState();
        }

        n++;
    }

    int num_sensing_steps = (n + 1) / (sensing_interval_ + 1);

    if (num_sensing_steps > 0)
        active_sensing_time_ = active_sensing_time / num_sensing_steps;
    else
        active_sensing_time_ = 0;
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
