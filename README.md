# REACT
## Using space debris as a distributed sensor for Earth's thermosphere.
### Repurposing historical satellite trajectory data for improved space safety and upper-atmosphere science

<img width="2368" height="1082" alt="image" src="https://github.com/user-attachments/assets/9729d976-934d-4cc4-bacb-e8a75aa01eee" />

The REACT framework (Response Estimation and Analysis using Correlated Trajectories) is a novel approach to space domain awareness and upper-atmosphere science that leverages publicly available satellite trajectory data (Two-line elements --- TLEs) to generate actionable insight into the operational state of low Earth orbit (LEO). Rather than treating each satellite as an isolated object, REACT learns the correlated dynamical response of thousands of satellites and debris objects to both natural forces (e.g., thermospheric density fluctuations from geomagnetic storms) and anthropogenic actions (e.g., propulsive satellite maneuvers).

At its core, REACT applies statistical estimation from historical correlations across the full tracked catalog of space objects in LEO. It uses “passive” debris populations as a distributed sensor to infer a consensus drag response to space weather, which provides a robust baseline against which individual satellites can be compared. Objects deviating significantly from this baseline can be flagged as potentially maneuvering, improving transparency in satellite operations and enabling more reliable collision-avoidance planning.

From a scientific perspective, REACT transforms decades of historical TLE data into a high-resolution record of thermospheric density as a function of latitude, longitude, and time. These reconstructions provide unprecedented coverage of the upper atmosphere, supporting studies of solar-terrestrial coupling, greenhouse gas–driven thermospheric cooling, and the impacts of space weather events on the orbital environment. The TLE dataset spans more than 50 years, and thus provides a long-term atmospheric specification for consistent comparison between space weather events across the space age. 

# Key Methods
Global Correlation Modeling: Captures dynamical correlations between satellites, using debris objects as reference “sensors.”

Gaussian Conditioning: Produces uncertainty-aware estimates of orbital decay rates and propagations by combining space weather drivers with observed satellite dynamics.

Maneuver/Anomaly Detection: Identifies likely maneuvers when satellite dynamics diverge from passive population (i.e. debris) responses.

Coordinate-Descent Density Reconstruction: Infers neutral density fields (ρ₀, H, q) directly from TLE-derived orbital decay rates.


# Use Cases
Transparency in Satellite Operations: Enables third-party monitoring of maneuvering behavior using publicly available data, increasing trust and accountability in LEO.

Collision Risk Management: Provides improved capability for satellite tracking and trajectory predictions during geomagnetic storms, reducing false conjunction alerts and unnecessary maneuvers.

Thermosphere Research: Offers a long-term, spatially resolved dataset for studying space weather impacts, atmospheric tides, and climate-driven density trends.

Policy and Sustainability: Supports analyses of orbital carrying capacity by quantifying how drag variability and conjunction rates change with solar cycle conditions.
