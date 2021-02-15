# 0.Abstract

​	This book  provides a survey on the state-of-the-art datasets and teniques. This survey includes both the historically most relevant literature as well as the current state of the art on several specific topics, including recognition, reconstruction, motion estimation, tracking, scene understanding, and  end-to-end learning for autonomous driving

​	A website that allows navigating topics as well as methods and provides additional information:

# 1.Introduction

* Q: Why fully autonomous navigation in general environments has not been realized to date
  - autonomous systems which operate in complex dynamic environments require models which generalize to unpredictable situations and reason in a timely manner.
  - informed decisions（正确的决定） require accurate perception, yet most of the existing computer vision models are still inferior to human perception and reasoning(推理？)

* Existing approaches to self-driving
  * modular pipelines
  * monolithic end-to-end learning approaches

## modular pipelines 

#### Brief Intro

The modular pipeline is the standard approach to autonoumous driving, mostly followed in the industry

#### Key Idea

- ​	Break down the complex mapping function from high-dimensional inputs to low-dimensional control variables into modules which can be independently developed, trained, and tested.
- Typical example of modularizing a self-driving stack
  -  Low-level perception+scene parsing+path planning+vehicle control
  - Or leverage machine learning(e.g.,deep neural networks) to extract low-level features or to parse the scene into individual components.
  - Path planning and vehicle control are dominated by classical state machines（状态机）, search algorithms, and control models.

#### Major Advantage

* They deploy human interpretable（可解析的） intermediate（中间的） representations such as detected objects or ==free space information== which allow gaining insights into failure modes of the system.
* The development of modular pipelines can be easily parallelized within companies where typically different team work on different aspects of the driving problem simultaneously.
* It's comparably easy to integrate ==first principle== and prior knowledge ( such as traffic laws or vehicle dynamics) about the problem into the system, <u>other aspects that are more difficult to specify by hand, suchas the appearance of pedestrians, are learned from large annotated datasets</u>

#### Major Drawback

* ==Human-designed intermediate representations are not necessarily optimal for the driving task==, which typically includes aspects like safety, comfort, and time for reaching the goal
* Most modules are trained and validated independently from each other, making use of auxiliary loss function. It's difficult to define appropriate intermediate representations and auxiliary loss function. (Take object detection as an example)

## end-to-end learning-based models

#### Goal

learn a function from observations to actions using a generic model such as a deep neural network

#### Learning approaches

* imitation learning 
  * suffers from overfitting and does not easily generalize to novel scenarios
* reinforcement learning
  * suffer from ==credit assignment== (信度分配）and ==reward shaping problems==（奖励函数设计）
  * typically slow and can only be applied in non-safety-critical simulation environment
* hoslistic neural network
  * ​	hard to interpret(black boxes)

## Related papers

H. Zhu, K. V. Yuen, L. Mihaylova, and H. Leung. “Overview of Environment Perception for Intelligent Vehicles”. In: *IEEE Trans. on Intelligent Transportation Systems (TITS)* PP.99 (2017), pp. 1–18.

## An interactive online tool(not accessible)

1 http://www.cvlibs.net/projects/autonomous_vision_survey

# 2.History of Autonomous Driving

* First demonstration of a driverless vehicle: "American Wonder"

  * time:	1925
  * developer:	Houdina Radio Control
  * performance:	a remote controlled vehicle that traveled along Broadway in New York City trailed byan operator in another vehicle

* Vision of a kind of adio-controlled electric cars that navigated via electromagnetic circuits installed in the roadway

  * time: 1939
  * prototypes
    * GM Firebird Ⅱ in 1956
    * RCA Labs' wire controlled car in 1960
    * ....
  * common points
    * depend on infrastructure
    * largely restricted to specific use cases

* First self-driving car prototype wich did not rely on dedicated infrastructure（专用基础设输）

  * time:	1986
  * team:	Navlab team at CMU and Ernst Dickmanns’s team at the Bundeswehr University Munich
  * Navlab1

* ALVINN

  * machanism
    * An imitation learning approach using a relatively small neutral network
  * team:	Navlab team at CMU
  * time: 	1989 or 1995?
  * performance:	Washington, D.C., to San Diego, CA, 98% automonously with manual longitudinal control(纵向控制)

* In the context of European PROMETHEUS project

  * team:	Dickmanns's team
  * mechanism:
    * A modular approach in which a vehicle and road model was used for continuously estimating the state and controlling of the vehicle
  * time:	1995
  * performance:	from Munich to Odense at velocities up to 175km/h with about 95% autonomous driving
  * scenario:	highway

* A real-time vision system for autonoumous driving in complex urban traffic situation

  * time:	
  * team:	Franke et al.
  * functions of the system:	
    * idepth-based obstacle detection 
    * tracking from stereo 
    * a framework for monocular detection and recognition of relevant objects such as traffic signs
  * conclusion
    * sensor capbilities should be enhanced
    * legal aspects should be considered carefully
    * the automation will likely be restricted to special infrastructures and will be
      extended gradually

* driver assistance system

  * first LIDAR-based distance control in 1995 by Mitsubishi
  * Radar-assisted adaptive cruise control(自适应巡航控制) in 1999 by Mercedes-Benz
  * Navigation systems and digital road maps became available in 2000
  * Today, differential GPS in combination with inertial measurement units allows for localization at an accuracy of 5cm in good conditions, enabling the use of detailed lane-level road maps(HD maps) and providing redundancy for noisy vision-based localization.

* DARPA：Defense Advanced Research Projects Agency

  * time：2004(Darpa Grand Challenge)、2005、2007(Darpa Urban Challenege)
  * In third race, most of the successful teams relied heavily on the emerging multi-beam LiDAR technology developed in a pioneering effort by Velodyne. 

* Google's self-driving program

  * time：2009
  * team members：Google took the lead and hired a range of star scientists who
    had participated in the Darpa Challenges
  * goal：develop a new driving platform and a custom, affordable multi-beam LiDAR scanner

* VisLab Intercontinental Autonomous Challenge

  * time：2010
  * sponsor：the VisLab team led by Alberto Broggi at the University of Parma in Italy 
  * Aim：drive semi-autonoumously from Parma in Italy to Shanghai in China
  * Detail：
    * a s econd vehicle automatically followed a route defined by a manually driven lead vehicle either visually or based on GPS waypoints sent by the lead vehicle.
    * The onboard system allowed for detecting obstacles, lane marking, ditches, berms , and to identify the presence and position of the preceding vehicle.

* Stadtpilot

  * team：Technical University of Braunschweig
  * time：2010
  * performance：able to navigate in a small geofenced innercity（地理围栏内城） area based on LIDAR, cameras, and HD maps.

* PROUD

  * team：VisLab
  * performance：a demonstration of inner-city and freeway driving in Parma

* the Grand Cooperative Driving Challenge

  * organizer：TNO
  * Aim：focusing on automonous cooperative driving behavior.During the competition, the semi-autonomous vehicles had to negotiate convoys , join convoys, and lead convoys. While longitudinal control was autonomous, lateral control was provided by a human safety driver. 

* KITTI Vision Benchmark

  * time：2012
  * significance：researchers around the globe were able to evaluate their progress on various self-driving perception tasks (including reconstruction, motion estimation, and object recognition) in a fair and objective manner. 

* S500 Intelligent Drive：a 103km autonomous ride on the historic Bertha Benz route from Mannheim to Pforzheim in Germany.

  * time：2013
  * sponsor：Mercedes Benz
  * team：Daimler research in collaboration with Karlsruhe Institute of Technology(KIT)
  * equipment：close-to-production sensor hardware、radar、camera
  * modules
    * Object detection and free-space analysis：using radar and stereo vision
    * Traffic light detection and object classification：monocular vision
    * centimeter-accurate localization relative manually annotated HD map：
      * point-feature-based
      * lane-marking based
  * performance：While focusing on a single route, the effort demonstrated that autonomous driving in complex inner-city environments based on close-to-production hardware and HD maps is feasible

* V-Charge

  * team：Volkswagen、Bosch、and other academic partner
  * Aim：a fully autonomous charging and parking of electric vehicles
  * performance：
    * a fully operational system has been demonstrated which included vision-only localization, mapping, navigation and control. 
    * supported many publications on different problems

* classification of automonous driving system

* ![image-20210215220150930](D:\ProgramData\notes\综述\Computer Vision for Autonomous Driving\images\image-20210215220150930.png)

  * level 0 (no autonomy) to level 5 (full autonomy)
  * time：2014

* S Class

  * time：2014
  * producer：Mercedes
  * level：level2（the driver has to monitor system at all times)

* Autopulot

  * time：2014
  * Tesla
  * level2
  * functions：autonomous steering, lane keeping, acceleration, and braking on
    the highway

* Tesla equipped all vehicles with eight cameras, twelve ultrasonic sensors, and a forward-facing radar from 2016

* Waymo

  * In 2016, after completing over 1,5 million miles, Google’s self-driving efforts became Waymo, a stand-alone subsidiary of Alphab et Inc. 

* NVIDIA‘s work

  * time：2016
  * using a single convolutional neural network

* last-mile delivery projects

  * a fully-eletric delivery system designed to safely get packages to Amazon customers using autonomous delivery devices

  

  









​	