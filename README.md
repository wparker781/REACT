# REACT
## Using space debris as a distributed sensor for Earth's thermosphere.

<img width="2368" height="1082" alt="image" src="https://github.com/user-attachments/assets/9729d976-934d-4cc4-bacb-e8a75aa01eee" />

The REACT framework (Response Estimation and Analysis using Correlated Trajectories) is an innovative approach to space domain awareness and upper-atmosphere science. It repurposes publicly available satellite trajectory data — specifically Two-Line Elements (TLEs) from the U.S. satellite catalog at [Space-Track.org](space-track.org) — to extract actionable insight into the health and behavior of low Earth orbit (LEO). Instead of treating each satellite as an independent object, REACT models the collective dynamical response of thousands of satellites and debris objects. This approach captures how the population as a whole reacts to natural forces, such as thermospheric density changes during geomagnetic storms, as well as human-driven actions, such as propulsive maneuvers.
Although TLEs were originally intended to provide near-real-time orbital knowledge for communications and tracking, their historical record across the entire satellite population is a powerful, underused data source. REACT uses this record to uncover patterns in satellite operator behavior—information that is rarely shared publicly—and to map changes in Earth’s upper atmosphere over time, creating a long-term, data-driven view of the thermosphere’s response to space weather.

At its core, REACT applies statistical estimation from historical correlations across the full tracked catalog of space objects in LEO. It uses “passive” debris populations as a distributed sensor to infer a consensus drag response to space weather, which provides a robust baseline against which individual satellites can be compared. Objects deviating significantly from this baseline can be flagged as potentially maneuvering, improving transparency in satellite operations and enabling more reliable collision-avoidance planning.

From a scientific perspective, REACT transforms decades of historical TLE data into a high-resolution record of thermospheric density as a function of latitude, longitude, and time. These reconstructions provide unprecedented coverage of the upper atmosphere, supporting studies of solar-terrestrial coupling, greenhouse gas–driven thermospheric cooling, and the impacts of space weather events on the orbital environment. The TLE dataset spans more than 50 years, and thus provides a long-term atmospheric specification for consistent comparison between space weather events across the space age. 

# Key Methods
**Global Correlation Modeling:** Captures dynamical correlations between satellites, using debris objects as reference “sensors.”

**Gaussian Conditioning:** Produces uncertainty-aware estimates of orbital decay rates and propagations by combining space weather drivers with observed satellite dynamics.

**Maneuver/Anomaly Detection:** Identifies likely maneuvers when satellite dynamics diverge from passive population (i.e. debris) responses.

**Coordinate-Descent Density Reconstruction:** Infers neutral density fields (ρ₀, H, q) directly from TLE-derived orbital decay rates.


# Use Cases
**Transparency in Satellite Operations:** Enables third-party monitoring of maneuvering behavior using publicly available data, increasing trust and accountability in LEO.

**Collision Risk Assessment:** Provides improved capability for satellite tracking and trajectory predictions during geomagnetic storms, reducing false conjunction alerts and unnecessary maneuvers.

**Thermosphere Research:** Offers a long-term, spatially resolved dataset for studying space weather impacts, atmospheric tides, and climate-driven density trends.

**Policy and Sustainability:** Supports analyses of orbital carrying capacity by quantifying how drag variability and conjunction rates change with solar cycle conditions.

# REACT in the News

REACT’s insights have been featured in leading journals and major media outlets, highlighting its impact on space sustainability and transparency:

- **Greenhouse Gas–Driven Thermospheric Contraction**  
  *Nature Sustainability* (2025)  
  [📄 Read the paper](https://www.nature.com/articles/s41893-025-01512-0) | [📰 Forbes coverage](https://www.forbes.com/sites/brucedorminey/2025/03/12/climate-change-is-even-wreaking-havoc-on-satellites-in-low-earth-orbit) | [📰 MIT News coverage](https://news.mit.edu/2025/study-climate-change-will-reduce-number-satellites-safely-orbit-space-0310) | [Github](https://github.com/ARCLab-MIT/ghg_kessler_capacity) 
  > CO₂ emissions are shrinking the upper atmosphere, reducing drag and **lowering the long-term satellite carrying capacity of LEO**.

- **Gannon Geomagnetic Storm & Satellite Mass Migration**  
  *Journal of Spacecraft and Rockets* (2024)  
  [📄 Read the paper](https://arc.aiaa.org/doi/10.2514/1.A36164) | [📰 Space.com coverage](https://www.space.com/may-solar-storm-largest-mass-migration-satellites)| [📰 Space News coverage](https://spacenews.com/geomagnetic-storms-cause-mass-migrations-of-satellites/)  
  > Documented the **largest satellite mass migration in history** during a major geomagnetic storm, exposing vulnerabilities in atmospheric drag forecasting and collision risk assessment.
