//
// Created by tipakorng on 3/3/16.
//

#include <visualization_msgs/MarkerArray.h>
#include "simulator.h"
#include "active_sensing_continuous_local/ObsrvBack.h"
#include "active_sensing_continuous_local/ReqObsrv.h"
#include "active_sensing_continuous_local/UpdateInfo.h"
#include "/home/user/row_ac_local/src/active_sensing_local-master/lib/random/include/rng.h"

unsigned int sensing_action_global;
double observation_global;
int req_obversation_flag = 0;
int update_flag = 0;
double update_location_x;
double update_location_y;
double update_location_z;
int communication_count = 0;
//int globalcheckflag = 0;
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

void reqObservationCallback(const active_sensing_continuous_local::ReqObsrv& msg)
{
 // ROS_INFO("observation: [%d]", msg.observe_there);
  sensing_action_global = msg.observe_there;
  req_obversation_flag = 1;
  //ROS_INFO("flag is %d", req_obversation_flag);
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
	ROS_INFO("communication round is %d",communication_count);
	ROS_INFO("n is %d",n);
	//ROS_INFO("sensing_interval is %d",sensing_interval_);
        //ROS_INFO("num_steps is %d",num_steps);
        // If sensing is allowed in this step.
        if (n % (sensing_interval_ + 1) == 0)
        {

            /*
                code below use spin loops to set timer that enable synchronization
            */

            ROS_INFO("REST FOR A WHILE!!!!!!!!");
            int counter_z = 0;
            while(counter_z < 100){
                counter_z++;
                ros::spinOnce();
            }
	
	 ros::Duration(0.4).sleep();
            ROS_INFO("REST DONE");

            /*
                code above use spin loops to set timer that enable synchronization
            */

            // Get sensing action.
            active_sensing_start = std::chrono::high_resolution_clock::now();
            sensing_action = planner_.getSensingAction();

            /*
                code below get sensing action    
            */

            ros::NodeHandle n_h;
            // ros::Subscriber sub = n_h.subscribe("req_obsrv", 1000, reqObservationCallback);
            ros::Rate loop_rate(10);
            // ROS_INFO("create a subber");

            // while(true){
            //     ros::spinOnce();
            //     loop_rate.sleep();
            //     if(req_obversation_flag == 1){
            //         break;
            //     }
            // }
            ROS_INFO("berfore waiting reqobsrv");
while(true){
            boost::shared_ptr<const active_sensing_continuous_local::ReqObsrv> obsrvmsg = ros::topic::waitForMessage<active_sensing_continuous_local::ReqObsrv>("req_obsrv", n_h, ros::Duration(0.4));
            active_sensing_continuous_local::ReqObsrv this_obsrv_msg;
            if(obsrvmsg != NULL){
                this_obsrv_msg = *obsrvmsg;
		 ROS_INFO("obsrvmsg is not null");
            sensing_action = this_obsrv_msg.observe_there;
		break;
            }
//}
            req_obversation_flag = 0;
           // ROS_INFO("waiting for observation request, req_obversation_flag is %d", req_obversation_flag);

}
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
            ROS_INFO("get into while(ros::ok()) loop obsrv_back");
            while (ros::ok())
            {
                int connections = observation_pub.getNumSubscribers();
                ROS_INFO("connected: %d", connections);
                active_sensing_continuous_local::ObsrvBack msg;
                msg.observation_back = observation_global;
                //ROS_INFO("observe back: %f", observation_global);
                if(connections > 0){
                    int i = 0;
                    while(i < 10){
                        observation_pub.publish(msg);
                        i++;
                        ros::spinOnce();
                    }

                ROS_INFO("published obsrv_back");
                break;
                }
		else
		{
           		 ros::Publisher observation_pub = n_h.advertise<active_sensing_continuous_local::ObsrvBack>("obversation_back", 1000);
		}
                loop_rate.sleep();
            }
            ROS_INFO("get out of while(ros::ok()) loop");

            /*
                code above publish observation
            */

            // //Eigen::IOFormat CommaInitFmt(4, 0, ", ", ", ", "", "", " << ", ";");
            // //std::cout << "observation is " << observation.format(CommaInitFmt) << std::endl;

            planner_.updateBelief(sensing_action, observation);

            // // Predict the new belief.
            task_action = planner_.getTaskAction();


            /*
                code below use spin loops to set timer that enable synchronization
            */

            ROS_INFO("REST FOR A WHILE!!!!!!!!");
            int counter_b = 0;
            while(counter_b < 100){
                counter_b++;
                ros::spinOnce();
            }
		    ros::Duration(0.4).sleep();
            ROS_INFO("REST DONE");
	

            /*
                code above use spin loops to set timer that enable synchronization
            */


            // /*
            //     code below get tast action
            // */

            // // ros::Subscriber sub3 = n_h.subscribe("update", 1000, updateCallback);
            // // ROS_INFO("create a subber3");

            // // while(true){
            // //     ros::spinOnce();
            // //     loop_rate.sleep();
            // //     if(update_flag == 1){
            // //         break;
            // //     }
            // // }
            // // ROS_INFO("get out of subber3 loop");

            // // update_flag = 0;
            // // ROS_INFO("update_flag is %d", update_flag);
            // // task_action(0) = update_location_x;
            // // task_action(1) = update_location_y;
            // // task_action(2) = update_location_z;

            ROS_INFO("waiting for update msg");
            boost::shared_ptr<const active_sensing_continuous_local::UpdateInfo> updatemsg = ros::topic::waitForMessage<active_sensing_continuous_local::UpdateInfo>("update", n_h, ros::Duration(10));
            active_sensing_continuous_local::UpdateInfo this_msg;
            if(updatemsg != NULL){
                this_msg = *updatemsg;
            }
            update_flag = 0;
            ROS_INFO("update message recieved");

            task_action(0) = this_msg.x;
            task_action(1) = this_msg.y;
            task_action(2) = this_msg.z;
// ROS_INFO("task_action(0) is %f",task_action(0));
            //ROS_INFO("task_action(1) is %f",task_action(1));
            //ROS_INFO("task_action(2) is %f",task_action(2));

            /*
                code above get tast action
            */
	ROS_INFO("statesback0 before predict is %f",states_.back()(0));
	ROS_INFO("statesback1 before predice is %f",states_.back()(1));

	    ROS_INFO("task_action(0) is %f",task_action(0));
            ROS_INFO("task_action(1) is %f",task_action(1));
            ROS_INFO("task_action(2) is %f",task_action(2));
            planner_.predictBelief(task_action);

            //std::cout << "task_action is " << task_action.format(CommaInitFmt) << std::endl;
	ROS_INFO("statesback0 before update is %f",states_.back()(0));
	ROS_INFO("statesback1 before update is %f",states_.back()(1));

	

            updateSimulator(sensing_action, observation, task_action);
	ROS_INFO("statesback0 after update is %f",states_.back()(0));
	ROS_INFO("statesback1 after update is %f",states_.back()(1));


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
	 ros::Duration(0.4).sleep();
            ROS_INFO("REST DONE");

            /*
                code above use spin loops to set timer that enable synchronization
            */


            /*
                code below get tast action
            */

            // ros::Subscriber sub4 = n_h.subscribe("updateelse", 1000, updateCallback);
            // ROS_INFO("create a subber4");

            // while(true){
            //     ros::spinOnce();
            //     loop_rate.sleep();
            //     if(update_flag == 1){
            //         break;
            //     }
            // }
            // ROS_INFO("get out of subber4 loop");
            
            ROS_INFO("waiting for updateelse msg");
            boost::shared_ptr<const active_sensing_continuous_local::UpdateInfo> updatemsg = ros::topic::waitForMessage<active_sensing_continuous_local::UpdateInfo>("updateelse", n_h, ros::Duration(10));
            active_sensing_continuous_local::UpdateInfo this_msg;
            if(updatemsg != NULL){
                this_msg = *updatemsg;
            }
            update_flag = 0;
            ROS_INFO("updateelse data recieved");

            task_action(0) = this_msg.x;
            task_action(1) = this_msg.y;
            task_action(2) = this_msg.z;

            /*
                code above get tast action
            */

	   // ROS_INFO("task_action(0) is %f",task_action(0));
          //  ROS_INFO("task_action(1) is %f",task_action(1));
           // ROS_INFO("task_action(2) is %f",task_action(2));
	ROS_INFO("statesback0 before predict is %f",states_.back()(0));
	ROS_INFO("statesback1 before predice is %f",states_.back()(1));

            planner_.predictBelief(task_action);
	ROS_INFO("statesback0 before update is %f",states_.back()(0));
	ROS_INFO("statesback1 before update is %f",states_.back()(1));
            updateSimulator(task_action);
	ROS_INFO("statesback0 after update is %f",states_.back()(0));
	ROS_INFO("statesback1 after update is %f",states_.back()(1));

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
	communication_count++;
	//ROS_INFO("statesback is %f",states_.back());
	ROS_INFO("!model_.isTerminal(states_.back()) after loop is %d",!model_.isTerminal(states_.back()));
	ROS_INFO("\n");
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
