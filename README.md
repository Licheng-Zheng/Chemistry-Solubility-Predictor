## Chemistry Solubility Predictor

### Purpose 
This project was my CAS project, which is a project that takes over a month, is collaborative and greatly contributes to my personal growth. This project was inspired by my friend introducing me to computational chemistry programs, greatly increasing my interest for thsi field that I had previously known nothing about. I was introduced to softwares such as GAMESS, ORCA and MOPAC, and I wanted to one day create a similar program. 

These projects are capable of predictor a humongous range of chemical properties that I would not be able to do without far more time, knowledge and effort that I had available to me, so for this project, I will only be creating a model that attempts to predict the solubility of a chemical, given its chemical formula. 

A longer project that this project was inspired by was my appreciation and passion for reinforcement learning, which I believe will become an integral part of future AI innovation. The issue with reinforcement modes, however, is that they require a model that closely resembles the environment that they will be found in the real world. For example, in order to create a robot that is efficient at sorting packages, it would be necessary to create a virtual environemnet that models that sorting environment as best as possible. This is due to the fact that if all trials were run on real machines, the costs and time required would make the project infeasible. By creating an  accurate chemistry model, I believe it will provide reinforcement llearning algorithms an accurate description of what will happen in the situation, allowing the algorithm to do whatever it needs to do. 

### Overview
For this project, I used a big pdf containing a large amount of solubility data, which I first processed into a list of 120 elements (the number of chemical elements I think, each represent an element, the other 2 are the temperature and solubility that the solubility in mols was expressed in). This is also the reason for a large issue that I will get into later on in this file. After processing into these lists, I ran a neural network on it in order to predict solubility based on chemical formula. I attempted to ue multiple optimization algorithms to experiment with the that would provide me the maximum accuracy. 

#### More Specific Information
The data that I used for this project was just a big pdf of a bunch of chemical compounds and their formulas at different temperatures. 
![image](https://github.com/user-attachments/assets/84acd6a5-12e6-491f-970e-d3deade4a992)
This information had to be processed before I would be able to use it, it would be made into lists of numbers of different compounds. THis would make it easy to feed to a neural network in the future. The issue with this approach was it was unable to keep track of isomers, which are compounds with the same molecular formula (meaning they contain the same amount of the same atoms), but arranged in a different manner. This different arrangement has a large effect on the solubility of a substance, and by not being able to consider it with my current approach, a lot of accuracy was sacrificed by using this method. 
![image](https://github.com/user-attachments/assets/64f535b9-c61e-4d36-acb0-b2d749150793)

### Next Steps 
1. Change it over, or implement a new solution that examines the IUPAC naming of the compound isntead of the molecular formula. The IUPAC formula tells us about structure as well as the molecular formula, and would provide much needed information on the structure of the molecule.
2. This should have been considered far earlier on in the process but something that I had not done that could have increased accuracy is normalize the data that is being provided to the algorithm. This is is an issue because data that is not normalized is harder or impossible to learn from. Which significantly decreases model accuracy. 
3. Rather than using a model that would try to predict the solubility of the compound, I could create more models that try to predict the bond lenght and strength between atoms. This could be done in a simulation of the molecule, to a point where it is the most stable. This was inspired by this chart that was seen on ORCA. The energy (and instability, less energy is more stable) is gradually reduced as teh program examines different positions for the molecule, until it finally finds the most stable molecule with the least amount of energy. However, this is a much harder asect of the project and will likely be something that I will only be able to get to in the future. 
![image](https://github.com/user-attachments/assets/93db3008-5db6-473a-b4a4-5aa65fc18c82)

