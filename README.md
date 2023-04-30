<a href="url"><img src="https://github.com/qcswat/qatrah/blob/main/img/Qatrah-logos.jpeg" align="left" height="75" width="75" ></a>

# QatraH | قطرة | (Droplet)

_Using quantum computing to design a more precise, environmental friendly and robust water distribution network and debugging._

[![License](https://img.shields.io/github/license/Qiskit/qiskit-terra.svg?style=popout-square)](https://www.gnu.org/licenses/) [![](https://img.shields.io/github/release/qcswat/qatrah.svg?style=popout-square)](https://pypi.org/project/qatrah/0.1.0/)
[![](https://img.shields.io/pypi/dm/qatrah.svg?style=popout-square)](https://pypi.org/project/qatrah/)

[QatraH Website Link](https://qcswat.github.io/qatrah/)

### NYUAD Hackathon for Social Good in the Arab World: Focusing on Quantum Computing (QC) and UN Sustainable Development Goals (SDGs).

https://nyuad.nyu.edu/en/events/2023/april/nyuad-hackathon-event.html

## Presentation

_The Slides can be viewed at [information](https://www.canva.com/design/DAFheTueBTE/iKTIXVJDEhtiVuoBL80QyA/view)._

## Motivation

![alt text](https://github.com/qcswat/qatrah/blob/main/img/water.gif)

Solving quantum solution for water based distibution and debugging using quantum computing.
Buzz words: WDN, Quantum Computing, QUBO, QML, Optimization, Pennylane, Jax

**Quantum algorithm**:

- Variational Quantum Eigensolver (VQE)
- Quantum Approximate Optimization Algorithm (QAOA)
- Quantum Machine Optimization Algorithm
- Quantum Machine Learning on Graph

**Installation Instructions:**
**_Requirements:_**
pip3 install -r requirements.txt

\*Conda users, please make sure to `conda install pip` and use it for the above requirements.

**Input to the program:**

- Sensor readings
- Paths from source to sensor
- Dataset of the water sample like from DEWA site under paywall

<p align="center">
  <img width="460" height="460" src="https://github.com/qcswat/qatrah/blob/main/WDN_20animation.gif">
</p>

## Replacing Classical Pressure Sensors with Optimized Quantum Sensors

Compared to classical pressure sensors, quantum sensors are not invasive. They are also tolerant to the changes in the environment around it while also being more accurate. This improves the ability to detect pipe leakage.

## Leak Detection and Localization

Leakage in water distribution systems has been a challenge for water management committees. The real-life data collected from the optimal placed sensors, can be used to predict and localise leakage by identifying the deviations of pressure in the network. This task can be done both using QUBO and Quantum Machine Learning based models.

### Using Quantum Machine Learning

Existing classical literature, suggests the use of machine learning to predict leakage and localise it to a particular pipe using the data from pressure sensors in the WDN at any given point of time. We attempt to solve the same using a quantum machine learning based model.

Specifically, we collect the pressure data from the optimally-placed sensors in a water distribution network to predict leakage in the WDN using a quantum neural network. It is implemented in the Pennylane framework using Jax. The data is fed into the model using Angle encoding. The model is composed of a parametrised quantum circuit with RY, RZ and CNOT gates which are trained over a total of 500 epochs. We use a train to test-set ratio of 4:1 and optimise the model using Rectified adam over the binary cross-entropy loss. At the end we obtain a test accuracy of 87.02% over the dataset of size 650.

## Acknowledgements

**Hackers:**
[Anas](https://github.com/AnasMM19), [Basant](https://github.com/Basant-Elhussein), [Mohammed](https://github.com/Mouhamedaminegarrach), [Airin](https://github.com/Rainiko66), [Lakshika](https://github.com/rathilakshika), [Sanjana](https://github.com/Sanjana-Nambiar), [Selin Doga](https://github.com/selindoga), [Yaser](https://github.com/YaserAlOsh)

**Mentors:**
[Fouad](https://github.com/fo-ui), [El Amine](https://github.com/qdevpsi3), [Victory Omole](https://github.com/vtomole), [Akash Kant](https://github.com/akashkthkr)

And thank you to the oragnising committee of NYUAD 2023 Hackathon https://nyuad.nyu.edu/en/events/2023/april/nyuad-hackathon-event.html and Qbraid and other student who made it possible and great.
