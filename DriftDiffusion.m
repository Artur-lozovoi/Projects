% Solving a system of master equations describing the number of electrons, holes, negatively and neutrally charged Nitrogen Vacancy centers, and 
% negatively and neutrally charged substitutional Nitrogen defects as a function of position on a 2D plane and time.
% 
% written by Artur Lozovoi
%
% Used in the manuscript entitled: "Probing metastable space-charge potentials in a wide bandgap semiconductor"
% by Lozovoi, A. et al. publsihed in Physical Review Letters 125, 256602 (2020).
% 
% Scaling factor 10^-7 is applied to some of the parameters (concentrations, diffusion coefficients, electric fields in cosistency with
% physical scaling laws of the problem. This trick allows convergence of the system of equations for high external elctric fields. Otherwise, 
% the PDEsolver tolerance does not suffice for obtaining smooth solutions

clear all; 

global E SC ppm Q P kN0 nvi Pi kB ec T un up dn1 dp1 dn dp parkX parkY
global Me v CCSp CCSn s NVnCapture NVpCapture kp NnCapture NpCapture gn gp Ip Ir Ic
 
ppm = 1.77*10^(5) * 10^(-7);      %  (*Number of defects per um^-3*)
Q = 0.01 * ppm;                   %  (*Overall concentration of nitrogen vacancies (NVs) in ppm*)
P = 1 * ppm;                      %  (*Overall concentration of substitutional nitrogen (N) in ppm*)
nvi = 0.7 * Q;                    %  (*Initial NV- concentration*)   
Pi = P - 0.7 * Q;                 %  (*Initial N0 density*)

kB = 1.380650 * 10^(-23);         %  (*Boltzmann constant J/K*)
ec = 1.602176 * 10^(-19);         %  (*Electronic charge C*)
Me = 9.11 * 10^(-31);             %  (*Electron mass in kg*)
T = 295;                          %  (*Temperature in K*)
v = (kB * T / Me)^(0.5) * 10^(6); %  (*Electron thermal velocity at temperature T in um/s*) 
un = 2.150 * 10^(11);             %  (*Electron eletrcical mobility in um^2/Vs*)   
up = 1.700 * 10^(11);             %  (*Hole eletrcical mobility in um^2/Vs*)
dn1 = (un * kB * T / ec);         %  (*Electron diffusion constant in um^2/s*)
dp1 = (up * kB * T / ec);         %  (*Hole diffusion constant in um^2/s*)
dn = dn1 * 10^(-7);               %  (*Scaled electron diffusion constant in um^2/s*)
dp = dp1 * 10^(-7);               %  (*Scaled hole diffusion constant in um^2/s*)

kN0 =  1;                         %  (*N0 ionization rate at the center of the beam in Hz*)
CCSp = 1.4 * 10^(-8);             %  (*N+ electron capture cross section in um^(2) from Pan et al "Carrier density dependent photoconductivity in diamond" APL (1990) *)
CCSn = 3.1 * 10^(-6);             %  (*N0 hole capture cross section in um^(2) from Pan et al "Carrier density dependent photoconductivity in diamond" APL (1990) *)
NnCapture = CCSn * v;             %  (*N electron volume capture rate in um^3*Hz *)
NpCapture = CCSp * v;             %  (*N hole volume capture rate in um^3*Hz *)
gn = NnCapture;                   %  (*Relative electron capture rate by N+ in Hz*)
gp = NpCapture;                   %  (*Relative hole capture rate by N in Hz*)  
NVnCapture = 10 * 10^(-5);        %  (*NV volume capture rates in um^3GHz*)
NVpCapture = 20 * 10^(-5);        %  (*NV volume capture rates in um^3GHz*)
kp = NVpCapture * 10^9;           %  (*Relative hole capture rate by NV- in Hz*)

Ip = 500 * 10^(-6) ;              %  (*Laser illumination instensity in W*)
Ir = 1.0 * 10^(-6) ;              %  (*Reference intensity in W*)
Ic= (Ip / Ir);                    %  (*Relative intensity*)
s = 1.5;                          %  (*Beam radius in um*)
E = 40 * 10^9 * 10^(-7);          %  (*Exteranal electric field strength in V/m*)
SC = 0.05 * 10^7;                 %  (*Space charge amplitude coefficient in V*m^2*)

t = [0 1e-3 5e-3 10e-3];          %  (*Time vector*)
timelength = length(t);           %  (*Number of sampled time points*)

% Below the system of parabolic partial differential equations for 6
% variables is defined in the following format:
% m * ?2u/?t2 + d * ?u/?t ? ?·(c * ?u) + a * u = f
% The system of 6 rquations is used:
% 1. Holes  2. Electrons  3. N0    4. NV-   5. x gradient of the charged defects   6. y gradient of the charged defects
% Equations 5 and 6 solve for auxillary variables that help introduce the space charge electric field, which is proportional to the second derivative of the charge distribution
% at each location
% In general, the terms describing the processes of photoionization and -recombination, carrier diffusion, drift and capture are included. The details of each term can be changed
% independently

model = createpde(6);                                             %  Creating a system of 6 PDEs
lengthX = 60;                                                     %  x dimension of the rectangular area
heightY = 40;                                                     %  y dimension of the rectangular area
R1 = [3, 4, 0, lengthX, lengthX, 0, 0, 0, heightY, heightY]';     %  Geometry defeinistion. First 2 rows, 3 and 4, indicate a rectangular geometry, next 4 rows x coordinates, last 4 are y coordinates
g = decsg(R1);                                                    %  Decomposing the defined geometry into minimal segments
geometryFromEdges(model,g);                                       %  Input of the geometry into the model
parkX = lengthX * 2 / 3;                                          %  Laser illumination x coordinate
parkY = heightY / 2;                                              %  Laser illumination y coordinate
c1 = [dp, dp, dn, dn, 0, 0, 0, 0, 0, 0, 0, 0]';                   %  Vector defining c coefficients in the PDE
d1 = [1, 1, 1, 1, 0, 0]';                                         %  Vector defining d coefficients in the DPE

% Definition of Dirichlet-type boundary conditions for each variable
applyBoundaryCondition(model,'dirichlet','Edge',1:4,'h',[1,0,0,0,0,0; 0,1,0,0,0,0; 0,0,1,0,0,0; 0,0,0,1,0,0; 0,0,0,0,1,0; 0,0,0,0,0,1],'r',[0,0,Pi,nvi,0,0]);

% Definition of coefficients in the PDE system
specifyCoefficients(model,'m',0,...
                          'd',d1,...
                          'c',c1,...
                          'a',@a2coeffunction,...
                          'f',@f2coeffunction);
                      
generateMesh(model,'Hmin',1,'Hmax',1,'GeometricOrder','linear');  %  Generate mesh with a scale size of 1  
setInitialConditions(model, [0, 0, Pi, nvi, 0, 0]')               %  Definition of the initial conditions       
results = solvepde(model, t);                                     %  Solution matrix
v = linspace(0, lengthX, 401);
[X , Y] = meshgrid(v, v / lengthX * heightY);                     %  Generate mesh for 2D solution plot
querypoints = [X(:), Y(:)]';
uintrp = interpolateSolution(results, querypoints, 4, timelength);%  Interpolate solution for 4th variable (NVs) for the last time point over query points
uintrp = reshape(uintrp, size(X));                                %  Rescaling the matrix
figure();
imagesc(v, v / lengthX * heightY, uintrp / Q)                     %  Plotting the 2D map of the scaled NV distribution
colormap(jet);

%  Below are the functions specifying the a and f coefficients that encode most of the terms in the PDEs

function a2matrix = a2coeffunction(location,state)

global SC gp gn kp k1 kh kN P Ic s kN0 parkX parkY

%  The terms below define photoionization rates of each defect as a function of the spatial position relative to the beam center according to the Gaussian profile.
%  The power scaling is either linear or quadratic depending on whether 1 or 2 photons are involved
%  (square(state.time*10000)/2+0.5) term commented out in the ionization rate definitions allows to probe 
%  time-dependent ionization, which demands higher computational power

k1 =  Ic*Ic*0.0046 * exp(-(((location.x-parkX).^2)+(location.y-parkY).^2)/(s^2)); % .* (square(state.time*10000)/2+0.5) ;         % (*Photoionization rate of NV-*)
kh =  Ic*Ic*0.0107 * exp(-(((location.x-parkX).^2)+(location.y-parkY).^2)/(s^2)); % .* (square(state.time*10000)/2+0.5) ;         % (*Photoionization rate of NV0*)
kN =  Ic*kN0 * exp(-(((location.x-parkX).^2)+(location.y-parkY).^2)/(s^2)); %.* (square(state.time*10000)/2+0.5) ;                % (*Photoionization rate of N0*)

N = 6;                      % Number of equations
nr = length(location.x);    % Number of columns
a2matrix = zeros(N,nr);     % Allocate a matrix

a2matrix(1,:) = gp * state.u(3,:) + kp * state.u(4,:) + SC * (state.uy(6,:) + state.ux(5,:)) ;
a2matrix(2,:) = gn * (P - state.u(3,:)) -  SC * (state.uy(6,:) + state.ux(5,:));
a2matrix(3,:) = kN + gn * state.u(2,:) + gp * state.u(1,:);
a2matrix(4,:) = k1 + kh  + kp * state.u(1,:);
a2matrix(5,:) = 1;
a2matrix(6,:) = 1;
end

function f2matrix = f2coeffunction(location,state)

global E SC gn P Q Ic s kN0 parkX parkY

%  (square(state.time*10000)/2+0.5) term commented out in the ionization rate definitions allows to probe 
%  time-dependent ionization, which demands higher computational power

k1 =  Ic*Ic*0.0046 * exp(-(((location.x-parkX).^2)+(location.y-parkY).^2)/(s^2)); % .* (square(state.time*10000)/2+0.5) ;         % (*Photoionization rate of NV-*)
kh =  Ic*Ic*0.0107 * exp(-(((location.x-parkX).^2)+(location.y-parkY).^2)/(s^2)); % .* (square(state.time*10000)/2+0.5) ;         % (*Photoionization rate of NV0*)
kN =  Ic*kN0 * exp(-(((location.x-parkX).^2)+(location.y-parkY).^2)/(s^2)); % .* (square(state.time*10000)/2+0.5) ;               % (*Photoionization rate of N0*)

N = 6;                      % Number of equations
nr = length(location.x);    % Number of columns
f2matrix = zeros(N,nr);     % Allocate f matrix

f2matrix(1,:) = kh .* (Q-state.u(4,:)) + state.ux(1,:) * E - SC * state.ux(1,:) .* state.u(5,:) - SC * state.uy(1,:) .* state.u(6,:);
f2matrix(2,:) = kN .* state.u(3,:) + k1 .* state.u(4,:) - state.ux(2,:) * E + SC * state.ux(2,:) .* state.u(5,:) + SC * state.uy(2,:) .* state.u(6,:);
f2matrix(3,:) = gn * P * state.u(2,:); 
f2matrix(4,:) = kh * Q; 
f2matrix(5,:) = state.ux(3,:) + state.ux(4,:);
f2matrix(6,:) = state.uy(3,:) + state.uy(4,:);

disp(state.time)            % Reflects status of the calcuation
end
