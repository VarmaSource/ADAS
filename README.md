# ADAS

##Introduction

Real-time object detection is an indispensable capability for ensuring the safety and advancement of autonomous vehicles (AVs) and advanced driver-assistance systems (ADAS). This project investigates the application of YOLOv5, a cutting-edge deep learning object detection algorithm, specifically for vehicle detection within autonomous driving environments. Object detection plays a critical role in safeguarding road users. By enabling AVs to identify and react to surrounding elements, such as pedestrians, traffic signals, and other vehicles, this technology significantly enhances road safety. Convolutional Neural Networks (CNNs) form the bedrock of this technology, facilitating real-time object recognition through the analysis of data acquired from cameras and sensors.

Despite the progress made in AV development, challenges remain. This project addresses the need for more comprehensive obstacle detection beyond traditional objects. Our focus lies on identifying uneven road surfaces, specifically potholes and humps, which pose significant safety hazards. By integrating YOLOv5 with a camera and an infrared sensor, the system gains the capability to detect these potential threats. Upon detection, the system can trigger a pre-programmed response, such as issuing a driver alert or initiating vehicle slowdown, ultimately promoting safer on-road navigation.

This project focuses on using YOLOv5, a cutting-edge object detection algorithm, for the specific task of identifying vehicles. We aim to demonstrate YOLOv5's effectiveness in this area and explore how it can improve autonomous driving. By equipping autonomous vehicles with the ability to detect uneven road surfaces through YOLOv5, we can significantly enhance their safety and overall perception, ultimately leading to more robust self-driving cars.


##Chapter 2 

Basic Concepts

This section contains the basic concepts about the related tools and techniques used in this project. 
2.1 Tools Used:

2.1.1 Python: 
Python is a popular programming language used for machine learning projects due to its simplicity and extensive range of libraries and frameworks. Python libraries like TensorFlow, Keras, PyTorch, and Scikit-learn are commonly used for image classification tasks.

2.1.2 Jupyter Notebook:
Code, equations, and visualizations can all be created and shared in documents using the open-source web program Jupyter Notebook. It is a well-liked instrument for machine learning and data analysis tasks.

2.1.3 Deep Learning Frameworks: 
TensorFlow, Keras, and PyTorch are a few examples of deep learning frameworks that offer pre-built models and training and building tools for intricate neural networks. They are often employed in jobs involving picture categorization.

2.2 Techniques used:

2.2.1 Data collection:
The first step in building a machine learning model is to collect data. In the case of a waste classification system, this would involve gathering images of different types of waste, such as plastic bottles, aluminum cans, paper, and so on.

2.2.2 Data Pre-processing:
Once you have collected your data, you will need to pre-process it to prepare it for use in your machine learning model. This might involve tasks such as resizing images to a consistent size, converting images to grayscale, and normalizing pixel values.

2.2.3 Training data preparation: 
One must separate the data into training and validation sets before you can train your machine learning model. The validation set is used to assess the model's performance, whereas the training set teaches the model to identify patterns in the data.

2.2.4 Model selection:
For image categorization, a variety of machine learning models, such as support vector machines (SVMs), decision trees, and convolutional neural networks (CNNs), can be employed. The most popular kind of model for image classification applications is CNNs.

2.2.5 Model training:
Once you have selected a model, you will need to train it on your training data. This involves feeding the model input data and adjusting the model's weights and biases to minimize the difference between the model's predictions and the actual labels.

2.2.6 Hyperparameter Tuning:
The performance of many machine learning models may be enhanced by adjusting their hyperparameters. These may consist of the pace of learning, the quantity of layers in the model, and the quantity of neurons in every layer. In order to determine which hyperparameter combinations work best for your model, you must test several combinations of hyperparameters. 

2.2.7 Model evaluation
After training your model, you must assess how well it performs using your validation set. This can let you gauge how effectively your model generalizes to fresh, untested data.

2.3 Literature Review

There have been several image and video processing research projects using neural networks in recent years. In 2016, Redmon et al. Proposed a groundbreaking work by  introducing the YOLO algorithm, which revolutionizes real-time object detection by proposing an end-to-end framework for predicting bounding boxes and class probabilities directly from images. This single-pass approach significantly enhances speed and accuracy, making it suitable for applications like speed breaker detection on roads.

In 2017 ,Bains et. al propose a real-time speed bump detection system based on Convolutional Neural Networks (CNNs), designed to enhance road safety. By analyzing video feeds with CNNs, the system achieves high accuracy in identifying speed bumps, aiding drivers in promptly responding to road hazards.

In 2018,  Redmon et. al present YOLOv3, an improved version of the YOLO algorithm with features like multi-scale prediction and feature pyramid networks. YOLOv3 achieves even higher accuracy while maintaining real-time processing capabilities, making it a valuable tool for tasks such as speed bump detection on roads.

In 2018  El-Alfy et al. thoroughly explore automated traffic sign detection and recognition systems utilizing deep learning techniques. They scrutinize methodologies, architectures, and challenges associated with real-world deployment, with a focus on the application of convolutional neural networks (CNNs) and recurrent neural networks (RNNs) in traffic sign recognition. The paper offers valuable insights for improving the reliability and scalability of automated traffic sign recognition systems.

In 2019, Alhamid et al. conduct a comprehensive review focusing on traffic sign detection and recognition techniques, particularly in the context of intelligent vehicles. Their survey covers traditional methods and recent advancements in deep learning, providing insights into performance metrics, datasets, and future research directions crucial for road safety.

In 2019, El Sallab et al. conduct a comprehensive review of object detection techniques tailored for autonomous vehicles, emphasizing real-world applications and challenges. Their analysis covers traditional methods and recent advancements, including the influential YOLO algorithm, providing valuable insights for autonomous driving systems.

In 2019 - Zou et. al present a comprehensive examination of deep learning techniques for object detection, covering various architectures, methodologies for training deep learning models, and evaluation metrics. Their review discusses the applications of object detection across diverse domains and identifies current challenges and future research directions, contributing significantly to the advancement of deep learning-based object detection methods.

In 2019, Kaur and Singh present a system for road traffic sign detection utilizing the YOLO object detection framework. Their work outlines the architecture tailored for traffic sign detection, emphasizing improvements in localization accuracy and potential applications for enhancing road safety.

In 2020, Ghaith et.al proposed a real-time object detection system for autonomous driving, leveraging cameras due to their rich information and cost-effectiveness. They implemented the You Only Look Once (YOLO) algorithm using a Darknet-53 CNN to detect pedestrians, vehicles, trucks, and cyclists. Training utilized the Kitti dataset collected from public roads via vehicle-mounted cameras. The system demonstrated high accuracy compared to traditional approaches, presenting promising results for road safety enhancement.


In 2021, Kavitha R et al. proposed a real-time object detection system utilizing the YOLO algorithm implemented on a Raspberry Pi platform. The system aimed to facilitate autonomous vehicle navigation by detecting eight classes of objects: car, bus, truck, person, pothole, wetland, motorcycle, and traffic light. The integration of a screen onboard the vehicle allowed users to experience the real-time detection capabilities firsthand.

In this project we had picked two of the State of The Art algorithms to train our model.


##Chapter 3 

Problem Statement & Requirement Specifications

3.1 Problem Statement: 
Despite the remarkable strides witnessed in autonomous vehicle technology, the persisting challenges in achieving comprehensive obstacle detection extend beyond the confines of conventional objects. Uneven road surfaces, ranging from potholes to humps, stand out as formidable adversaries, posing substantial safety concerns that reverberate across the spectrum of road users and infrastructure integrity. It is within this intricate landscape that the limitations of traditional object detection algorithms become glaringly evident, failing to sufficiently grapple with the nuanced complexities of irregular road surfaces. These inadequacies not only jeopardize passenger safety but also cast shadows over the broader aspirations of autonomous driving technology, impeding its seamless integration into mainstream transportation systems. Consequently, the imperative to innovate and develop robust solutions tailored specifically for identifying and mitigating the hazards posed by uneven road surfaces becomes increasingly pronounced, heralding a new era of safety and efficiency in autonomous mobility solutions.

3.2 Requirement Specifications:
Detection Accuracy: The system must accurately identify vehicles and uneven road surfaces, including potholes and humps, with high precision and recall rates to ensure road safety.
Integration with Sensors: Integration with cameras and infrared sensors is necessary to capture data for object detection in various environmental conditions, including low light and adverse weather.
Real-time Performance: The object detection system should operate in real-time to provide timely responses to potential hazards, minimizing the risk of accidents.
Scalability: The system should be salable to accommodate different vehicle types, road conditions, and environments, ensuring its effectiveness across various scenarios.
3.3 Project Planning:
Research and Literature Review: Conduct an in-depth review of existing object detection algorithms and research related to autonomous driving and road safety.
Data Collection and Annotation: Gather a diverse dataset of vehicle and road surface images annotated with labels for training and evaluation purposes.
Model Development: Develop and train YOLOv5 models tailored for vehicle detection and uneven road surface identification using the annotated dataset.
Integration and Testing: Integrate the trained models with camera and infrared sensor systems, and conduct extensive testing in simulated and real-world driving scenarios.
Evaluation and Optimization: Evaluate the performance of the system in terms of detection accuracy, real-time performance, and robustness, and optimize as necessary to meet project requirements.

3.4 Project Analysis:
The risk analysis for the project involves identifying potential project risks and their impact on the project. The risks may include technical risks such as the accuracy and efficiency of the machine learning algorithm, the quality of the training data set, and the integration of the system with different hardware and software platforms. The risks may also include project management risks such as delays in project timelines, inadequate resource allocation, and budget overruns. 
To mitigate the risks, the project team may implement risk management strategies such as regular project reviews, continuous testing and evaluation of the system, and contingency planning. Additionally, the team may establish clear communication channels with the Project Guide to ensure that potential risks are identified and addressed in a timely manner.

3.5 System Design:

3.5.1 System Architecture: 
The system architecture consists of the following components:
Input Module: Captures data from cameras and infrared sensors.
Preprocessing Module: Preprocesses the input data, including image enhancement and feature extraction.
Object Detection Module: Utilizes YOLOv5 for vehicle detection and uneven road surface identification.
Decision Making Module: Analyzes detected objects and triggers appropriate responses, such as driver alerts or vehicle maneuvers.
Output Module: Communicates with the vehicle's control systems to implement the desired actions based on the detected objects.

3.5.2 Design Constraints:
Computational Resources: The system must operate within the computational constraints of the onboard hardware in autonomous vehicles, ensuring efficient utilization of resources.
Power Consumption: Minimize power consumption to prolong the vehicle's battery life and optimize energy efficiency.
Cost: Consider the cost implications of hardware components and software development, aiming for a cost-effective solution that can be feasibly deployed in commercial autonomous vehicles.


##Chapter 4

Implementation

The system is equipped with minimal sensors and equipment, comprising an Infrared Camera for real-time video capture and an Infrared sensor for distance calculation between the vehicle and objects. The camera is strategically positioned at an angle to ensure optimal video capture without interference, while the Infrared sensor is mounted at an altitude conducive to accurate detection of humps and potholes.
The camera activates when the vehicle exceeds 30 km/h for over 200 seconds and deactivates after the vehicle remains stationary for more than 40 seconds. Upon activation, the infrared sensor and relay module are also activated. The relay module engages when the vehicle's speed exceeds predefined parameters and the distance between the vehicle and objects falls below a predefined threshold, ensuring judicious application of automated braking to prevent unnecessary slowdowns.
The relay module interfaces with a compact motor capable of ejecting fluid, without affecting the vehicle's primary braking system. The fluid reservoir remains untouched, ensuring user comfort and compactness. The automated braking is selectively applied to the rear wheel, allowing the user to retain control of the front brake in emergency situations.

4.1Methodology

4.1.1 Real-Time Video Capture:
The system employs an Infrared camera strategically positioned to capture real-time video data when the vehicle surpasses 30 km/h for over 200 seconds. This activation threshold ensures that video recording commences when the vehicle is in motion for an extended period, enhancing the system's responsiveness to dynamic driving conditions and potential hazards.

4.1.2 Object Detection using YOLOv5 and CNN:
For real-time object detection, the system utilizes the YOLOv5 algorithm due to its remarkable efficiency and accuracy. YOLOv5 processes the incoming video stream, leveraging Convolutional Neural Networks (CNNs) integrated within its architecture. These CNNs play a pivotal role in rapidly and accurately identifying objects of interest by analyzing image data, enabling the system to detect various road features and potential obstacles with precision.

4.1.2.1 YOLOv5: 
YOLOv5, an iteration of the You Only Look Once (YOLO) object detection algorithm, stands at the forefront of real-time visual recognition. Leveraging a single-stage detection approach, YOLOv5 processes entire images in one pass, enabling rapid inference and high detection accuracy. Its architecture, built upon convolutional neural networks (CNNs), efficiently extracts features from input images, facilitating precise object identification across various scales and complexities. YOLOv5's versatility extends to fine-tuning pre-trained models for specific tasks, offering developers an accessible yet powerful solution for diverse applications such as autonomous driving and surveillance. Advanced features like feature pyramid networks (FPNs) and attention mechanisms further enhance its detection capabilities, ensuring robust performance in challenging environments. Overall, YOLOv5 represents a significant leap forward in object detection technology, empowering developers with an efficient and adaptable tool for real-time visual recognition tasks.

4.1.2.2 Convolution Neural Networks (CNNs):
Convolutional Neural Networks (CNNs) revolutionize image analysis and recognition tasks through their hierarchical feature extraction capabilities. Comprising convolutional, pooling, and fully connected layers, CNNs process input images to extract intricate features like edges, textures, and shapes. Convolutional layers apply convolution operations to input images, while pooling layers reduce spatial dimensions, preserving essential information. Fully connected layers integrate extracted features for classification tasks like object detection. CNNs' ability to learn patterns from raw data, coupled with training on large datasets, ensures robust performance in diverse applications, including object detection, image classification, and semantic segmentation. In essence, CNNs serve as foundational components in computer vision systems, driving advancements in fields ranging from autonomous driving to medical imaging.

4.1.3 Simultaneous Processes:
The system orchestrates two concurrent processes to ensure comprehensive monitoring of the vehicle's surroundings. Firstly, the Object Detection process identifies critical elements such as humps, potholes, and zebra crossings, while simultaneously calculating their distance from the vehicle. This information forms the basis for determining whether braking intervention is necessary to mitigate potential risks. Secondly, the Speed Limit Sign Detection process scans for and interprets speed limit signs, facilitating adherence to traffic regulations and enhancing overall road safety.

4.1.4 Object Detection and Braking:
Upon detecting relevant objects, the system's algorithm meticulously evaluates their proximity to the vehicle, taking into account predefined parameters for safe driving. If the calculated distance falls within a specified range warranting precautionary action, the system initiates braking interventions to mitigate the identified hazards effectively. This proactive approach ensures timely responses to potential dangers, minimizing the risk of accidents and enhancing passenger safety.

I.Speed Bump Detection Module:
Implement YOLO for its efficiency in real-time object detection, configuring it to detect and localize speed bumps accurately.
Integrate the module with onboard cameras and processing units, optimizing for low-latency performance.

II.Pothole Detection Module:
Develop a CNN-based model tailored for pothole detection, leveraging techniques such as transfer learning and data augmentation to improve performance.
Train the model using a diverse dataset of pothole images captured unde
under various conditions to enhance robustness.

III.Plain Road Detection Module:
Design a CNN architecture specialized in identifying segments of plain road, distinguishing them from areas with anomalies.
Ensure the module integrates seamlessly with the overall system to provide continuous feedback to the driver regarding road conditions.

4.1.5 Automated Braking Execution:
To execute automated braking maneuvers, the system interfaces with a relay module connected to a compact motor. When the vehicle's speed and proximity to detected objects align with predefined criteria indicative of potential hazards, the relay module activates the compact motor, triggering the automated braking process. This mechanism swiftly applies braking force as needed, allowing the vehicle to respond promptly to evolving road conditions and maintain a safe driving environment for all occupants.


##Chapter 5 

Standards Adopted

5.1Design Standards
Our project follows the design standards set forth by prominent organizations such as IEEE (Institute of Electrical and Electronics Engineers) and ISO (International Organization for Standardization). These standards provide a structured framework for project design, encompassing aspects such as system architecture, hardware designs, and block diagrams.

The project design adheres to the principles outlined in IEEE standards, incorporating clear and concise representations of the system architecture. Block diagrams have been employed to visually illustrate the various components and their interactions, ensuring a comprehensive understanding of the project's structure.
Additionally, the design standards help in maintaining consistency and clarity in the documentation, facilitating effective communication among project stakeholders.

5.2Coding Standards
Coding standards play a pivotal role in maintaining code quality, readability, and consistency throughout the development process. Our project embraces well- established coding standards to enhance the maintainability and collaborative aspects of the codebase. Some of the coding standards followed include:
Conciseness: Code is written with the goal of minimizing unnecessary lines, promoting clarity and efficiency.
Naming Conventions: Appropriate and consistent naming conventions are employed for variables, functions, and other code entities, ensuring a clear and understandable codebase.
Code Segmentation: Blocks of code within the same section are segmented into paragraphs, aiding readability and comprehension.
Indentation: Indentation is used to demarcate the beginning and end of control structures, enhancing the visual structure of the code.
Modularization: Lengthy functions are avoided, and a modular approach is adopted, where each function performs a specific and well-defined task.
These coding standards contribute to the overall code quality, making it more maintainable and accessible to developers working on the project.

5.3Testing Standards
To ensure the quality and reliability of the project, our team adheres to recognized testing standards established by ISO and IEEE. These standards provide guidelines for quality assurance and verification processes, contributing to the robustness of the developed solution. The testing standards include:
ISO Standards: Our project follows ISO standards related to software quality assurance and testing. These standards help in defining the criteria for successful project completion and the verification of desired outcomes.
IEEE Standards: We also adhere to relevant IEEE standards for testing, ensuring that the project meets industry-accepted benchmarks for quality and reliability.
By incorporating these testing standards, we aim to validate the functionality of our mobile app for streamlined attendance monitoring, providing a high-quality and dependable solution for end-users.


##Chapter 6

Conclusion and Future Scope
6.1Conclusion

This project has developed a robust system for real-time object detection in autonomous driving scenarios, focusing on identifying vehicles and hazardous road conditions like potholes and humps. Utilizing YOLOv5 and Convolutional Neural Networks (CNNs), the system can accurately detect objects even in challenging environments. By integrating with a camera and infrared sensor, it provides timely alerts and automated responses, enhancing road safety. The implementation and testing phases have demonstrated the system's effectiveness in detecting speed humps, potholes, and plain road segments with high accuracy, showcasing its potential to improve autonomous driving capabilities.

6.2Future Scope

Advanced object detection techniques, coupled with sensor fusion methods, offer opportunities to significantly enhance the system's accuracy and reliability in identifying road features and obstacles. Integration with Vehicle-to-Infrastructure (V2I) communication systems stands as a key frontier, enabling real-time information exchange and bolstering the system's ability to anticipate hazards effectively. Incorporating adaptive learning algorithms and artificial intelligence (AI) capabilities can empower the system to continuously evolve and adapt to changing driving conditions, enhancing its responsiveness and effectiveness. Augmented reality (AR) interfaces present another exciting avenue, offering drivers real-time visual feedback and alerts to improve situational awareness and safety on the road.
