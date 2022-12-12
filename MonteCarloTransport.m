% Statistical Monte Carlo calculation of hole transport and capture by an individual Coulombically attractive point-defect trap in diamond
% 
% written by Artur Lozovoi
%
% Used in a manuscript entitled: "Observation and Monte Carlo modeling of carrier capture at a single point defect under variable electric fields"
% by Lozovoi, A. et al. which is under review in Physical Review Letters

clear all

kB = 1.380650 * 10^(-23);         % Boltzmann constant in [J/K]
ec = 1.602176 * 10^(-19);         % Electronic charge in [C]
Me = 9.11 * 10^(-31);             % Electron mass in [kg]
Mh = Me * 0.9;                    % Heavy hole mass in [kg]
T = 10;                           % Temperature in [K] 
E0 = 8.85e-12;                    % Vacuum permitivity
E = 5.4;                          % Dielectric constant of diamond
h = 1.054e-34;                    % h-bar in [J*s]
pd = 3.515e3;                     % Diamond density in [kg/m^3]
v  = (3 * kB * T / Mh)^0.5;       % Hole thermal velocity in [m/s]
E1 = 12 * ec;                     % Deformation potential of diamond in [J]
ul = 12000;                       % Sound velocity in diamond in [m/s]
Eopt = 0.13 * ec;                 % Optical phonon energy in [eV]
%Dopt = 4e10 * ec;                % Optical intervalley scattering deformatiomn potential interaction //Suntornwipat et al Nano Lett 2021 % not used now
Dopt = 0.1 * 2.5e11 * ec;         % Optical phonon deformation potential in [eV]// Smajdi et al APL 2013
mu = 0.2;                         % Hole mobility in [m^2/s] 
a0 = 3.54e-10;                    % Diamond lattice constant in [m]

% Calculation of the hole effective mass that appears in the expression for optical phonon scattering probability and includes an integral over 
% polar and azimuthal angles
A = 3.61;                         %
B = 0.18;                         % Valence band warping parameters for diamond, Reggiani & Jacoboni review (Appendix)
C = 3.76;                         %
fun = @(psi, phi) sin(phi) .* (1 - ((B / A) ^ 2 + (C / A) ^ 2 * (sin(phi) .^ 4 .* sin(psi) .^ 2 .* cos(psi) .^ 2 + sin(phi) .^ 2 .* cos(phi) .^2)) .^0.5) .^ (-1.5);
I = integral2 (fun, 0, 2 * pi, 0, pi);                  
Mhopt = (Me ^ 1.5 / 4 / pi / A ^ 1.5 * I) ^ (2 / 3);  % Hole effective mass       

duration = 10e-9;                 % Duration of the simulation in [s]
Scat_time = 0.13e-12;             % Average scattering time in [s]
steps = duration/Scat_time*100;   % Number of time steps in the simulation
distance = 3000;                  % Starting position relative to the trap in [nm]
averages = 50000;                 % Number of repetitions of the statistical calculation
fields = 7;                       % Number of electroc fields values at which calculation is performed
section = zeros(fields,1);        % Allocate a vector for capture cross sections

Eh0 = Mh * v^2 / 2;
x0 = 2 * Mh ^ 0.5 * ul * Eh0 ^ 0.5 / kB / T;

I_ab_0 = 2 * x0 ^ 2 * (1 - 34 / 105 * 2 ^ 0.5 * x0 + x0 ^ 2 / 12) + 2 / 3 * kB * T / Eh0 * x0 ^ 3 * (136 * 2 ^ 0.5 / 105 - x0 + 44 * 2 ^ 0.5 * x0 ^ 2 / 315);
Pa0 = E1 ^ 2 * Mh ^ 0.5 * (kB * T) ^ 3 / 2 ^ 4.5 / pi / pd / ul ^ 4 / h ^ 4 * Eh0 ^ (-0.5) * I_ab_0;               
I_em_0 = 2 * x0 ^ 2 * (1 + 34 / 105 * 2 ^ 0.5 * x0 + x0 ^ 2 / 12) - 2 / 3 * kB * T / Eh0 * x0 ^ 3 * (136 * 2 ^ 0.5 / 105 + x0 + 44 * 2 ^ 0.5 / 315 * x0 ^ 2);
Pe0 = E1 ^ 2 * Mh ^ 0.5 * (kB * T) ^ 3 / 2 ^ 4.5 / pi / pd / ul ^ 4 / h ^ 4 * Eh0 ^ (-0.5) * I_em_0;

if Eh0 <= 2 * Mh * ul^2
     Pe0 = 0;
end
 
Popte0 = 0;
Popta0 = 3 * Dopt^2 * Mhopt^1.5 / 2^1.5 / pi / h^2  / Eopt / pd * (1 / (exp(Eopt / kB / T) - 1)) * (Eh0 + Eopt) ^ 0.5;            
            
P_00 = Pe0 + Pa0 + Popte0 + Popta0;
t_step = 1 / P_00 / 100;

parpool('local', 80);            % Parallelization for running on the cluster (Flatiron Institute)

for iii = 1:fields
    captured = 0;
    capturedNot = 0;
    close = 0;
    parfor ii = 1:averages
        Phi0 = rand * 2 *pi;
        Teta0 = rand * pi;
        vd = v;
        vx = vd * sin(Teta0) * cos(Phi0);
        vy = vd * sin(Teta0) * sin(Phi0);
        vz = vd * cos(Teta0);
        u0 = [0, vx, 0, vy, distance * 1e-9, vz];
        emis = 0;
        absorb = 0;
        emisOpt = 0;
        absorbOpt = 0;
        flagclose = 0;
        k = 0;
        t1 = 0;
        t2 = 0;
        while t2 < duration
            t1 = t2;
            r = rand;
            while r == 0 || r > 1 - 1e-20
                r = rand;
            end
            t2 = t1 - t_step * log(r);
            
            [t,u] = ode23(@(t,u) [u(2); -ec^2/Mh/E0/4/pi/E/(u(1)^2+u(3)^2+u(5)^2)^(3/2)*u(1); u(4); -ec^2/Mh/E0/4/pi/E/(u(1)^2+u(3)^2+u(5)^2)^(3/2)*u(3);  u(6); -ec^2/Mh/E0/4/pi/E/(u(1)^2+u(3)^2+u(5)^2)^(3/2)*u(5)], [t1 t2], u0);
            
            b = (u(end,2) ^ 2 + u(end,4) ^ 2 + u(end,6) ^ 2) ^ 0.5;
            radius = (u(end,1) ^ 2 + u(end,3) ^ 2 + u(end,5) ^ 2) ^ 0.5;
            
            if radius > 9000e-9
                capturedNot = capturedNot + 1;
                break
            end
            
            if b < 0
                break
            end
            
            Eh = Mh * b^2 / 2;
            Ep = - ec^2 / E0 / 4 / pi / E / radius;
            
            if (Eh + Ep) < -  7 * kB * T
                captured = captured + 1;
                break
            end
            
            if radius < 5e-9
                if flagclose == 0
                    close = close + 1;
                    flagclose = 1;
                end
            end
        
            x = 2 * Mh ^ 0.5 * ul * Eh ^ 0.5 / kB / T;
                     
            if x <= 3 / 2 ^ 0.5
                I_ab_1 = 2 * x ^ 2 * (1 - 34 / 105 * 2 ^ 0.5 * x + x ^ 2 / 12) + 2 / 3 * kB * T / Eh * x ^ 3 * (136 * 2 ^ 0.5 / 105 - x + 44 * 2 ^ 0.5 * x ^ 2 / 315);
                Pa = E1 ^ 2 * Mh ^ 0.5 * (kB * T) ^ 3 / 2 ^ 4.5 / pi / pd / ul ^ 4 / h ^ 4 * Eh ^ (-0.5) * I_ab_1;
                
                I_em_1 = 2 * x ^ 2 * (1 + 34 / 105 * 2 ^ 0.5 * x + x ^ 2 / 12) - 2 / 3 * kB * T / Eh * x ^ 3 * (136 * 2 ^ 0.5 / 105 + x + 44 * 2 ^ 0.5 / 315 * x ^ 2);
                Pe = E1 ^ 2 * Mh ^ 0.5 * (kB * T) ^ 3 / 2 ^ 4.5 / pi / pd / ul ^ 4 / h ^ 4 * Eh ^ (-0.5) * I_em_1;
            else
                I_ab_1 = 1017 / 280 + (68 - 2358 * x ^ (-2) + 41931 * x ^ (-4)) * exp(-3) - 8 * exp(-2^0.5 * x) * (x ^ 2 + 4 * 2^0.5 * x + 28 + 72 * 2^0.5 / x + 252 * x ^ (-2) + 270 * 2^0.5 * x ^ (-3) + 270 * x ^ (-4)) + 2 / 3 * kB * T / Eh * (801 / 140 + (312 - 13248 * x ^ (-2) + 300078 * x ^ (-4)) * exp(-3) - exp(-2 ^ 0.5 * x) * (8 * 2^0.5 * x ^ 3 + 72 * x ^ 2 + 288 * 2 ^ 0.5 * x + 1824 + 4320 * 2 ^ 0.5 / x + 14400 * x ^ (-2) + 15120 * 2 ^ 0.5 * x ^ (-3) + 15120 * x ^ (-4)));
                Pa = E1 ^ 2 * Mh ^ 0.5 * (kB * T) ^ 3 / 2 ^ 4.5 / pi / pd / ul ^ 4 / h ^ 4 * Eh ^ (-0.5) * I_ab_1;
                
                I_em_1 = 5913 / 280 + 136 * 2^0.5 / 105 * x ^ 3 - 9 * (4 - 162 / 5 * x ^ (-2) + 729 / 7 * x ^ (-4)) - 2 / 3 * kB * T / Eh * (6371 / 140 + 2 * x ^ 4 - 81 + 729 * x ^ (-2) - 19683 / 8 * x ^ (-4));
                Pe = E1 ^ 2 * Mh ^ 0.5 * (kB * T) ^ 3 / 2 ^ 4.5 / pi / pd / ul ^ 4 / h ^ 4 * Eh ^ (-0.5) * I_em_1;
            end
            
            if Eh <= 2 * Mh * ul^2
                Pe = 0;
            end
            
            Popte = 0;
            Popta = 3 * Dopt^2 * Mhopt^1.5 / 2^1.5 / pi / h^2  / Eopt / pd * (1 / (exp(Eopt / kB / T) - 1)) * (Eh + Eopt) ^ 0.5;            
            
            if b > v * 2.3
                Popte = 3 * Dopt^2 * Mhopt^1.5 / 2^1.5 / pi / h^2  / Eopt / pd * (1 / (exp(Eopt / kB / T) - 1) + 1) * (Eh - Eopt) ^ 0.5;
            end

            probability = rand / tstep;
            if probability < Popte
                scattering = 1;
            else
                if probability < Popte + Popta
                    scattering = 2;
                else
                    if probability < Popte + Popta + Pe
                        scattering = 3;
                    else
                        if probability < Popte + Popta + Pe + Pa
                            scattering = 4;
                        else
                            scattering = 0;
                        end
                    end
                end
            end
        
            switch scattering
                case 0                                         % no scattering             
                    b1 = b;
                    u0 = [u(end,1), u(end,2) , u(end,3),  u(end,4), u(end,5), u(end,6)];
                case 1                                         % Optical phonon emission
                    flagAngle = 0;
                    cosphi = u(end,6)/b;            % z
                    sinphi = (1 - cosphi^2)^0.5;
                    cospsi = u(end,2) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;     % xy
                    sinpsi = u(end,4) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;
                    while flagAngle ==0
                        r1 = rand;
                        cosphi1 = 1 - 2 * r1;
                        sinphi1 = (1 - cosphi1^2)^0.5;
                        r2 = rand;
                        cospsi1 = cos(2 * pi * r2);
                        sinpsi1 = sin(2 * pi * r2);
                        f1 = rand * 3 * E1 ^ 2 * h / 2 ^ 5 / pi ^ 2 / pd / Eopt / a0 ^ 2 * (1 / (exp(Eopt / kB / T) - 1) + 1) * (Eh - Eopt) ^ 0.5 / (h ^ 2 * A / 2 / Me * (1 - ((B / A) ^ 2 + (C / A) ^ 2 * 2.1) ^ 0.5)) ^ 1.5;
                        CC = 3 * E1 ^ 2 * h / 2 ^ 5 / pi ^ 2 / pd / Eopt / a0 ^ 2 * (1 / (exp(Eopt / kB / T) - 1) + 1) * (Eh - Eopt) ^ 0.5 / (h ^ 2 * A / 2 / Me * (1 - ((B / A) ^ 2 + (C / A) ^ 2 * (sinphi1 ^ 4 * sinpsi1 ^ 2 * cospsi1  ^2 + sinphi1 ^ 2 * cosphi1 ^ 2)) ^ 0.5)) ^ 1.5;
                        if f1 <= CC
                            flagAngle = 1;
                            cosphi11 = cosphi1;
                            sinphi11 = sinphi1;
                            cospsi11 = cospsi1;
                            sinpsi11 = sinpsi1;
                        end
                    end
                    b1 = b - 2.3 * v;
                    if b1 < 0
                        b1 = 0;
                        disp('Achtung1!')
                        disp(b)
                    end
                    emisOpt = emisOpt + 1;
                    u0 = [u(end,1), b1 * (sinphi11 * sinpsi11) , u(end,3),  b1 * (sinphi11 * cospsi11), u(end,5), b1 * cosphi11];
                
                case 2                                         % Optical phonon absorption
                    flagAngle = 0;
                    cosphi = u(end,6)/b;            % z
                    sinphi = (1 - cosphi^2)^0.5;
                    cospsi = u(end,2) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;     % xy
                    sinpsi = u(end,4) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;
                    while flagAngle ==0
                        r1 = rand;
                        cosphi1 = 1 - 2 *rand;
                        sinphi1 = (1 - cosphi1^2)^0.5;
                        r2 = rand;
                        cospsi1 = cos(2 * pi * r2);
                        sinpsi1 = sin(2 * pi * r2);
                        
                        f1 = rand * 3 * E1 ^ 2 * h / 2 ^ 5 / pi ^ 2 / pd / Eopt / a0 ^ 2 * (1 / (exp(Eopt / kB / T) - 1)) * (Eh + Eopt) ^ 0.5 / (h ^ 2 * A / 2 / Me * (1 - ((B / A) ^ 2 + (C / A) ^ 2 * 2.1) ^ 0.5)) ^ 1.5;
                        CC = 3 * E1 ^ 2 * h / 2 ^ 5 / pi ^ 2 / pd / Eopt / a0 ^ 2 * (1 / (exp(Eopt / kB / T) - 1)) * (Eh + Eopt) ^ 0.5 / (h ^ 2 * A / 2 / Me * (1 - ((B / A) ^ 2 + (C / A) ^ 2 * (sinphi1 ^ 4 * sinpsi1 ^ 2 * cospsi1  ^2 + sinphi1 ^ 2 * cosphi1 ^ 2)) ^ 0.5)) ^ 1.5;
                        if f1 <= CC
                            flagAngle = 1;
                            cosphi11 = cosphi1;
                            sinphi11 = sinphi1;
                            cospsi11 = cospsi1;
                            sinpsi11 = sinpsi1;
                        end 
                    end
                    b1 = b + 2.3 * v;
                    absorbOpt = absorbOpt + 1;
                    u0 = [u(end,1), b1 * (sinphi11 * sinpsi11) , u(end,3),  b1 * (sinphi11 * cospsi11), u(end,5), b1 * cosphi11];
                
                case 3                                         % Acoustic phonon emission
                    flagAngle = 0;
                    
                    cosphi = u(end,6) / b;            % z
                    sinphi = (1 - cosphi^2) ^ 0.5;
                    
                    cospsi = u(end,2) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;     % xy
                    sinpsi = u(end,4) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;
                    psi = angle(cospsi + 1i * sinpsi) + pi;
                    
                    while flagAngle ==0
                        r1 = rand;
                        cosphi1 = 1 - 2 * r1;
                        sinphi1 = (1 - cosphi1 ^ 2) ^ 0.5;
                        psi1 =  2 * pi *rand;
                        cospsi1 = cos(psi1);
                        sinpsi1 = sin(psi1);
                        
                        costeta = cosphi * cosphi1 + sinphi * sinphi1 * cos(psi - psi1);
                        x1 = Mh * b * 2 ^ 0.5 / h * (1 - costeta) ^ 0.5;
                        f1 = rand * Mh ^ 3 * E1 ^ 2 * 2 ^ 0.5 / pi ^ 2 / pd / ul / h ^ 4 * b ^ 2 * (1 / (exp(h * ul * (Mh * b * 2 / h) / kB / T) - 1) + 1) * (1 + 3 * 1 ^ 2) * (1 + 1) ^ 0.5 * (1 - h * ul * (Mh * b * 2 / h) / Mh / b ^ 2);
                        CC =  Mh ^ 3 * E1 ^ 2 * 2 ^ 0.5 / pi ^ 2 / pd / ul / h ^ 4 * b ^ 2 * (1 / (exp(h * ul * (x1) / kB / T) - 1) + 1) * (1 + 3 * costeta ^ 2) * (1 - costeta) ^ 0.5 * (1 - h * ul * (x1) / Mh / b ^ 2);
                        if f1 <= CC
                            qphonon = x1;
                            flagAngle = 1;
                            if (b ^ 2 - h * ul * qphonon * 2 / Mh) < 0
                                flagAngle = 0;
                            end
                        end
                    end
                    b1 = (b ^ 2 - h * ul * qphonon * 2 / Mh) ^ 0.5;
                    if (b ^ 2 - h * ul * qphonon * 2 / Mh) < 0
                        b1 = 0;
                        disp('Achtung1!')
                        disp(qphonon)
                        disp(b)
                    end
                    
                    emis = emis + 1;
                    u0 = [u(end,1), b1 * (sinphi1 * sinpsi1) , u(end,3),  b1 * (sinphi1 * cospsi1), u(end,5), b1 * cosphi1];
                
                case 4                                         % Acoustic phonon absorption
                    flagAngle = 0;
                    
                    cosphi = u(end,6) / b;            % z
                    sinphi = (1 - cosphi^2) ^ 0.5;
                    
                    cospsi = u(end,2) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;     % xy
                    sinpsi = u(end,4) / (u(end,4) ^ 2 + u(end,2) ^ 2) ^ 0.5;
                    psi = angle(cospsi + 1i * sinpsi) + pi;
                    
                    while flagAngle ==0
                        r1 = rand;
                        cosphi1 = 1 - 2 * r1;
                        sinphi1 = (1 - cosphi1^2)^0.5;
                        psi1 =  2 * pi *rand;
                        cospsi1 = cos(psi1);
                        sinpsi1 = sin(psi1);
                        
                        costeta = cosphi * cosphi1 + sinphi * sinphi1 * cos(psi - psi1);
                        x1 = Mh * b * 2^0.5 / h * (1 - costeta)^0.5;
                        f1 = rand * Mh^3 * E1^2 * 2^0.5 / pi^2 / pd / ul / h^4 * b^2 * (1 / (exp(h * ul * (Mh * b * 2 / h) / kB / T) - 1)) * (1 + 3 * 1^2) * (1 + 1)^0.5 * (1 + h * ul * (Mh * b * 2 / h) / Mh / b^2);
                        CC =  Mh^3 * E1^2 * 2^0.5 / pi^2 / pd / ul / h^4 * b^2 * (1 / (exp(h * ul * (x1) / kB / T) - 1) + 0) * (1 + 3 * costeta^2) * (1 - costeta)^0.5 * (1 + h * ul * (x1) / Mh / b^2);
                        if f1 <= CC
                            qphonon = x1;
                            flagAngle = 1;
                        end
                    end
                    b1 = (b ^ 2 + h * ul * qphonon * 2 / Mh) ^ 0.5;
                    absorb = absorb + 1;
                    u0 = [u(end,1), b1 * (sinphi1 * sinpsi1) , u(end,3),  b1 * (sinphi1 * cospsi1), u(end,5), b1 * cosphi1];
            end
            
            Eh = Mh * b1 ^ 2 / 2;
            Ep = - ec^2 / E0 / 4 / pi / E / radius;
            
            if (Eh + Ep) < - 7 * kB * T
                captured = captured + 1;
                break
            end
            k = k + 1;
        end
    end
    section(iii) = captured / (capturedNot + captured) * 4 * pi * distance ^ 2;
    disp(7-iii)
    disp(section(iii))
    disp(capturedNot+captured) 
end

% display the calculated capture cross sections
disp(section)
