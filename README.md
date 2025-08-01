# Enhanced-Adaptive-QoS-Framework-for-Multi-Tenant-SDN-with-Hybrid-ML-Driven-Prioritization
This project simulates a hybrid bus-star network architecture incorporating Software-Defined Networking (SDN) principles to manage dynamic packet routing and quality of service . The system compares traditional rule-based traffic classification with a machine learning approach, using real network traffic features to prioritize packets effectively.

| File                                                   | Description                                      |
| ------------------------------------------------------ | ------------------------------------------------ |
| `bus_star_hybrid_sim.py`                               | Main simulation with animation and QoS plots     |
| `ml_sdn_controller.py`                                 | Intelligent SDN controller using ML model        |
| `sdn_controller.py`                                    | Rule-based SDN controller                        |
| `packet_simulator.py`                                  | Generates synthetic packet data                  |
| `network_packet_simulation.py` / `star_network_sim.py` | Variants for simplified or star-only simulations |
| `ml_trianing.py`                                       | Trains and exports the ML model                  |
| `packet_data.csv`                                      | Dataset for training ML classifier               |
| `ml_priority_model.joblib`                             | Trained model used for real-time classification  |
