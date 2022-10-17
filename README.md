# Trajectory-prediction
Trajectory prediction in urban automated driving under unknown intention


Automated driving is required to plan for future actions in a dynamic and uncertain environment. Uncertainty
comes from predictions from uncertain intentions and measurement noise. Interacting multiple model Kalman filter in
automated driving scenarios enables accurate state estimation of dynamically changing objectives by using multiple models. The
IMM algorithm combines multiple models tracking results to update the state distribution of dynamical obstacles and update
the model probability at the next time point based on the model likelihood function. Non-linear model predictive control
as a ego vehicle controller is a strategy for determining optimal control which achieves the control objective under constraints.
We alternate multiple models with different probabilities for the dynamical obstacles, find the optimal strategy in dynamics with
NMPC and display the results in Matlab. Our approach can be used to avoid other vehicles and achieve the desired speed with
