function thesis_structural_robustness_validation_N2000()
% THESIS_STRUCTURAL_ROBUSTNESS_VALIDATION_N2000  Checks the weak-selection
% zero crossing when k, z or mu is varied about the baseline
% (k = 6, z = 0.1, mu = 1e-3), with R = 1, S = 0.2, P = -0.2 and T swept
% (MATLAB R2016a).
%
% For every case the coefficients A_mu, B_mu and the predicted crossing
% T* = R + (B_mu/A_mu)(S-P) are computed. T is swept over seven values in
% [T* - 0.1, T* + 0.1], and the numerical weak-selection slope (fitted
% through the origin over beta = 0.001, 0.002 and 0.004) is compared with
% the theoretical line. The horizontal axis of each panel is T - T*, so
% the predicted crossing is at zero. The result is Fig. C2 of the paper.
%
% Usage:
%   thesis_structural_robustness_validation_N2000
%
% Output file:
%   structural_robustness_validation_N2000.png  figure with three panels
%                                               (varying k, z and mu)
% For every case, x0_mu, A_mu, B_mu, sigma = A_mu/B_mu and the theoretical
% and numerical crossings are printed in the command window.
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

clc; close all;

%% -------------------- main settings --------------------
% Baseline structure; each panel changes one of k, z and mu
base.k  = 6;
base.z  = 0.10;
base.mu = 1e-3;

k_values  = [4 6 8];
z_values  = [0.05 0.10 0.20];
mu_values = [1e-4 1e-3 1e-2];

% Payoffs (T is swept)
game.R = 1.00;
game.S = 0.20;
game.P = -0.20;

% Small selection intensities, half-width of the T sweep and number of
% T values
beta_small = [0.001 0.002 0.004];
sweepSpan  = 0.10;
nSweep     = 7;

% Colour and marker of the three values in each panel
colors  = [0.000 0.447 0.741; ...
           0.850 0.325 0.098; ...
           0.000 0.550 0.180];
markers = {'o','s','d'};

fig = figure('Color','w','Position',[80 120 1650 650]);

fprintf('\nStructural robustness validation, N = 2000\n');
fprintf('Game: R = %.2f, S = %.2f, P = %.2f, swept payoff = T\n', ...
    game.R,game.S,game.P);

% One panel per varied quantity
make_panel(fig,1,k_values ,'k' ,base,game,beta_small,sweepSpan,nSweep,colors,markers);
make_panel(fig,2,z_values ,'z' ,base,game,beta_small,sweepSpan,nSweep,colors,markers);
make_panel(fig,3,mu_values,'mu',base,game,beta_small,sweepSpan,nSweep,colors,markers);

safe_save(fig,'structural_robustness_validation_N2000.png');

end

%% ==========================================================
function make_panel(fig,panel_id,values,mode,base,game,beta_small,sweepSpan,nSweep,colors,markers)
% Draw one panel. For each entry of 'values' (values of k, z or mu, as
% chosen by mode = 'k', 'z' or 'mu'), plot the theoretical slope (line)
% and the numerical slope (markers) against T - T*, and print the
% coefficients and crossings.

% Positions of the panels, their legends and the legend captions
% (normalized figure units)
panel_pos   = [0.060 0.170 0.205 0.700; ...
               0.320 0.170 0.205 0.700; ...
               0.580 0.170 0.205 0.700];
legend_pos  = [0.830 0.695 0.145 0.205; ...
               0.830 0.395 0.145 0.205; ...
               0.830 0.095 0.145 0.205];
caption_pos = [0.830 0.905 0.145 0.035; ...
               0.830 0.605 0.145 0.035; ...
               0.830 0.305 0.145 0.035];

ax = axes('Parent',fig,'Position',panel_pos(panel_id,:));
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
set(ax,'FontName','Times New Roman','FontSize',10,'LineWidth',1.0);

nV  = length(values);
h   = gobjects(2*nV,1);
leg = cell(2*nV,1);

for i = 1:nV
    if strcmp(mode,'k')
        p = make_par(values(i),base.z,base.mu);
        tag = sprintf('k = %d',p.k);       legtag = sprintf('%d',p.k);
    elseif strcmp(mode,'z')
        p = make_par(base.k,values(i),base.mu);
        tag = sprintf('z = %.2f',p.z);     legtag = sprintf('%.2f',p.z);
    else
        p = make_par(base.k,base.z,values(i));
        tag = sprintf('mu = %.0e',p.mu);   legtag = sprintf('%.0e',p.mu);
    end

    res = validate_one(p,game,beta_small,sweepSpan,nSweep);

    h(2*i-1) = plot(ax,res.xplot,res.theory,'-','Color',colors(i,:),'LineWidth',2.3);
    h(2*i)   = plot(ax,res.xplot,res.slope,'LineStyle','none','Marker',markers{i}, ...
        'MarkerSize',6,'MarkerEdgeColor',colors(i,:),'MarkerFaceColor','w','LineWidth',1.3);
    leg{2*i-1} = [legtag ' theory'];
    leg{2*i}   = [legtag ' numeric'];

    fprintf(['%-12s x0 = %.6f, A_mu = %.4e, B_mu = %.4e, sigma = %.4f, ' ...
             'T*_theory = %.6f, T*_numeric = %.6f, converged %d/%d\n'], ...
        tag,res.x0,res.Amu,res.Bmu,res.Amu/res.Bmu, ...
        res.crossTheory,res.crossNum,res.n_ok,res.n_runs);
end

% Zero line and predicted crossing (T - T* = 0)
xlim(ax,[-sweepSpan sweepSpan]);
xl = xlim(ax);
yl = ylim(ax);
plot(ax,xl,[0 0],'k:','LineWidth',1.1);
plot(ax,[0 0],yl,'--','Color',[0.35 0.35 0.35],'LineWidth',1.1);
xlim(ax,xl); ylim(ax,yl);

xlabel(ax,'T - T^*_{theory}','FontWeight','bold','Interpreter','tex','FontSize',20);
if panel_id == 1
    ylabel(ax,'Weak-selection slope','FontWeight','bold','FontSize',20);
end

if strcmp(mode,'k')
    title(ax,{'Degree robustness','Varying k'},'FontWeight','bold','FontSize',18);
    legend_caption = 'k';
elseif strcmp(mode,'z')
    title(ax,{'Zealot-fraction robustness','Varying z'},'FontWeight','bold','FontSize',18);
    legend_caption = 'z';
else
    title(ax,{'Mutation-rate robustness','Varying \mu'},'FontWeight','bold', ...
        'Interpreter','tex','FontSize',18);
    legend_caption = '\mu';
end

lgd = legend(h,leg,'Interpreter','tex','FontSize',14);
set(lgd,'Position',legend_pos(panel_id,:));

annotation(fig,'textbox',caption_pos(panel_id,:), ...
    'String',legend_caption,'EdgeColor','none', ...
    'HorizontalAlignment','center','FontWeight','bold', ...
    'FontSize',14,'Interpreter','tex');

end

%% ==========================================================
function res = validate_one(p,game,beta_small,sweepSpan,nSweep)
% Weak-selection test for one parameter set p. The structure res has
% the fields x0, Amu, Bmu, xplot (= T - T*), theory, slope, the
% theoretical and numerical crossings (crossTheory, crossNum), and the
% numbers of converged and attempted steady-state runs (n_ok, n_runs).

ws  = weak_selection_coefficients(p,0.50);
Amu = ws.A;
Bmu = ws.B;

% Predicted crossing, swept T values and theoretical slope
Tstar = game.R + (Bmu/Amu)*(game.S-game.P);
Tvec  = linspace(Tstar-sweepSpan,Tstar+sweepSpan,nSweep);

theory = Amu*(game.R-Tvec) + Bmu*(game.S-game.P);
slope  = NaN(1,nSweep);
n_ok   = 0;
nB     = length(beta_small);

% Numerical slope at each T: steady states for the small betas (each
% started from the previous one), fitted through the origin. Only runs
% that pass the residual test (method = 1) are used.
for i = 1:nSweep
    pG    = set_payoffs(p,game.R,game.S,Tvec(i),game.P);
    Yseed = ws.Y0;
    dx    = NaN(nB,1);
    for b = 1:nB
        [Yss,method] = solve_ss(beta_small(b),Yseed,pG);
        if method == 1
            n_ok  = n_ok + 1;
            dx(b) = Yss(1) - ws.x0;
            Yseed = Yss;
        end
    end
    slope(i) = fit_through_origin(beta_small,dx);
end

res.x0          = ws.x0;
res.Amu         = Amu;
res.Bmu         = Bmu;
res.xplot       = Tvec - Tstar;
res.theory      = theory;
res.slope       = slope;
res.crossTheory = zero_cross(Tvec,theory);
res.crossNum    = zero_cross(Tvec,slope);
res.n_ok        = n_ok;
res.n_runs      = nSweep*nB;

end

%% ==========================================================
function p = make_par(k,z,mu)
% Parameter structure for degree k, zealot fraction z and mutation rate
% mu, with the solver settings used for the weak-selection tests.

p.k  = k;
p.mu = mu;
p.N  = 2000;
p    = set_z(p,z);

p.rhs_tol    = 1e-10;
p.tail_tol   = 0;
p.bound_tol  = 1e-8;
p.allow_tail = false;
p.min_pass   = 1;
p.reltol     = 1e-9;
p.abstol     = 1e-11;
p.fdY        = 1e-6;
p.tiny       = 1e-14;
p.max_pass   = 40;
p.pass_time  = 50*p.N;
p.max_step   = 5*p.N;
p.coeff      = make_coeff(k);

end

%% ==========================================================
%  WEAK-SELECTION TOOLS
%% ==========================================================
function ws = weak_selection_coefficients(p,x_init)
% Neutral regularized equilibrium Y0 and structural coefficients
%   A_mu = -e1' J^{-1} U,   B_mu = -e1' J^{-1} V.
% x_init is the initial x used to find the neutral state. The output
% structure ws has the fields Y0, x0, res (residual), J, cond (condition
% number of J), A and B.

p0  = set_payoffs(p,0,0,0,0);          % payoffs are irrelevant at beta = 0
[Y0,method,res] = solve_ss(0,init_state(x_init,p0),p0);
if method ~= 1
    error(['Neutral steady state not converged (residual %.3e). ' ...
           'Increase max_pass or pass_time.'],res);
end
if Y0(1) <= 0 || Y0(1) >= 1
    error('Neutral steady state is not interior (x0 = %.3e).',Y0(1));
end

J     = num_jac(Y0,p0);
[U,V] = structural_vectors(Y0,p0);

ws.Y0   = Y0;
ws.x0   = Y0(1);
ws.res  = res;
ws.J    = J;
ws.cond = cond(J);
ws.A    = -[1 0 0]*(J\U);
ws.B    = -[1 0 0]*(J\V);

if ws.cond > 1e10
    warning('Neutral Jacobian is nearly singular (cond = %.3e).',ws.cond);
end

end

%% ==========================================================
function [U,V] = structural_vectors(Y0,p)
% Forcing vector b = (R-T) U + (S-P) V at the neutral equilibrium, where
% b is the derivative of the right-hand side with respect to beta at
% beta = 0. The neighbourhood averages are written in closed form
% (paper, Appendix A), with q0 = q_{D|D} and r0 = q_{D|C}.
% The third components carry (1-mu)/(4 k N), consistent with the 1/(kN)
% prefactor of the pZC equation.

k  = p.k;
mu = p.mu;
N  = p.N;
x   = Y0(1);
pCC = Y0(2);
pZC = Y0(3);

pC  = p.u*x;
pD  = p.u*(1-x);
pCD = max(pC-pCC-pZC,0);
pZD = max(p.z-p.pZZ-pZC,0);

[cC,zC,r0] = cond_probs(pCC,pZC,pC,p.tiny);   % around a cooperator
[cD,zD,q0] = cond_probs(pCD,pZD,pD,p.tiny);   % around a defector

ED_A2 = (1-q0)^2 + q0*(1-q0)/k;
ED_AB = q0*(1-q0)*(1-1/k);
EC_AB = r0*(1-r0)*(1-1/k);
EC_B2 = r0^2 + r0*(1-r0)/k;

Phi_q = k - ((k-1)*(2*k-1)/k)*q0 + ((k-1)*(k-2)/k)*q0^2;   % E_D[kC A^2]/cD
Psi_q = ((k-1)*q0/k)*((k-1)-(k-2)*q0);                     % E_D[kC AB]/cD
Psi_r = ((k-1)*r0/k)*((k-1)-(k-2)*r0);                     % E_C[kC AB]/cC
Omg_r = ((k-1)*r0/k)*(1+(k-2)*r0);                         % E_C[kC B^2]/cC

U = [ (1-mu)*((1-x)*ED_A2 + x*EC_AB)/(4*N); ...
      (1-mu)*(pD*cD*Phi_q + pC*cC*Psi_r)/(2*k*N); ...
      (1-mu)*(pD*zD*Phi_q + pC*zC*Psi_r)/(4*k*N) ];

V = [ (1-mu)*((1-x)*ED_AB + x*EC_B2)/(4*N); ...
      (1-mu)*(pD*cD*Psi_q + pC*cC*Omg_r)/(2*k*N); ...
      (1-mu)*(pD*zD*Psi_q + pC*zC*Omg_r)/(4*k*N) ];

end

%% ==========================================================
function J = num_jac(Y,p)
% Central-difference Jacobian of the scaled RHS (includes 1/N, like U, V).
% The perturbed states are projected onto the admissible region; the
% actual step Yp(j)-Ym(j) is used as the denominator.

n = length(Y);
J = zeros(n,n);
for j = 1:n
    h = p.fdY*max(1,abs(Y(j)));
    e = zeros(n,1);
    e(j) = 1;
    Yp  = sanitize(Y + h*e,p);
    Ym  = sanitize(Y - h*e,p);
    den = Yp(j) - Ym(j);
    if abs(den) < 1e-16
        error('Jacobian step collapsed for component %d (state on a boundary).',j);
    end
    J(:,j) = (rhs(Yp,0,p) - rhs(Ym,0,p))/den;
end

end

%% ==========================================================
function xc = zero_cross(x,y)
% First zero crossing of y(x) by linear interpolation (NaN if none).

xc = NaN;
for i = 1:length(x)-1
    if isnan(y(i)) || isnan(y(i+1))
        continue;
    end
    if y(i) == 0
        xc = x(i);
        return;
    elseif y(i)*y(i+1) < 0
        xc = x(i) - y(i)*(x(i+1)-x(i))/(y(i+1)-y(i));
        return;
    end
end

end

%% ==========================================================
function s = fit_through_origin(beta_vec,dx)
% Least-squares slope of dx = s*beta (dx = x*(beta) - x0).

beta_vec = beta_vec(:);
dx = dx(:);
ok = ~isnan(dx);
if ~any(ok)
    s = NaN;
else
    s = sum(beta_vec(ok).*dx(ok))/sum(beta_vec(ok).^2);
end

end

%% ==========================================================
%  MODEL CORE (identical in every script)
%
%  Pair approximation on a k-regular network with zealous cooperators
%  (Z), ordinary cooperators (C) and ordinary defectors (D). The state is
%  Y = [x; pCC; pZC]: x is the fraction of ordinary players who
%  cooperate, and pCC, pZC are ordered pair densities. The structure p
%  holds k, N, mu, the zealot data z, u = 1 - z and pZZ (set_z), the
%  payoffs R, S, T, P (set_payoffs), the multinomial coefficients coeff
%  (make_coeff) and the solver settings used by solve_ss.
%% ==========================================================
function F = rhs(Y,beta,p)
% Right-hand side of the ODE system, with the 1/N prefactor written out:
%   dY/dt = F_raw(Y)/N.

F = rhs_raw(Y,beta,p)/p.N;

end

%% ==========================================================
function F = rhs_raw(Y,beta,p)
% Unscaled right-hand side N*dY/dt. Used for all residual checks.
%
%   x'    = (1-x) T+  - x T-
%   pCC'  = (2/k) [ pD G+_CC - pC G-_CC ]
%   pZC'  = (1/k) [ pD G+_ZC - pC G-_ZC ]
%
% T+ (T-) is the probability that a randomly chosen ordinary defector
% (cooperator) switches its strategy. G+_CC (G-_CC) is the expected
% number of C-C links created (removed) by that update, and G+_ZC,
% G-_ZC are the same for Z-C links. All four are averages over the
% neighbourhood compositions (kZ, kC, kD) with multinomial weights
% (pair-approximation closure).
%
% The pCC equation has prefactor 2/k and the pZC equation 1/k: a C-C
% link contributes two ordered pairs to pCC (one in each direction),
% whereas a Z-C link contributes one ordered pair to pZC (and one to
% pCZ).

Y   = sanitize(Y,p);
x   = Y(1);
pCC = Y(2);
pZC = Y(3);
k   = p.k;
mu  = p.mu;

% Node densities and the remaining pair densities (conservation of
% pairs: pCD = pC - pCC - pZC, pZD = z - pZZ - pZC)
pC  = p.u*x;
pD  = p.u*(1-x);
pCD = max(pC - pCC - pZC,0);
pZD = max(p.z - p.pZZ - pZC,0);

% Probabilities of the neighbour types (C, Z, D) around an ordinary
% cooperator (q*_C) and around an ordinary defector (q*_D)
[qC_C,qZ_C,qD_C] = cond_probs(pCC,pZC,pC,p.tiny);
[qC_D,qZ_D,qD_D] = cond_probs(pCD,pZD,pD,p.tiny);

% Sum over all neighbourhoods with kZ zealots, kC ordinary cooperators
% and kD ordinary defectors; c*q^kZ*q^kC*q^kD is the multinomial weight
Tplus  = 0;  Tminus = 0;
GCCp   = 0;  GCCm   = 0;
GZCp   = 0;  GZCm   = 0;

for kZ = 0:k
    for kC = 0:(k-kZ)
        kD = k - kZ - kC;
        c  = p.coeff(kZ+1,kC+1);

        A = (kZ+kC)/k;                 % fraction of cooperative neighbours
        B = kD/k;                      % fraction of defective neighbours
        Delta = A*(p.R-p.T) + B*(p.S-p.P);

        phip = mu + (1-mu)*A*fermi( beta*Delta);   % D -> C
        phim = mu + (1-mu)*B*fermi(-beta*Delta);   % C -> D

        % Contributions of a focal defector (D -> C) and of a focal
        % cooperator (C -> D) with this neighbourhood
        probD = c*(qZ_D^kZ)*(qC_D^kC)*(qD_D^kD);
        Tplus = Tplus + probD*phip;
        GCCp  = GCCp  + probD*kC*phip;
        GZCp  = GZCp  + probD*kZ*phip;

        probC  = c*(qZ_C^kZ)*(qC_C^kC)*(qD_C^kD);
        Tminus = Tminus + probC*phim;
        GCCm   = GCCm   + probC*kC*phim;
        GZCm   = GZCm   + probC*kZ*phim;
    end
end

% Unscaled right-hand sides of x, pCC and pZC
F = [ (1-x)*Tplus - x*Tminus; ...
      2*(pD*GCCp - pC*GCCm)/k; ...
      (pD*GZCp - pC*GZCm)/k ];

end

%% ==========================================================
function y = fermi(a)
% Numerically stable logistic function 1/(1+exp(-a)).

if a >= 0
    y = 1/(1+exp(-a));
else
    ea = exp(a);
    y = ea/(1+ea);
end

end

%% ==========================================================
function [qC,qZ,qD] = cond_probs(pairC,pairZ,total,tiny)
% Conditional neighbour probabilities around a focal node of density
% 'total'. pairC and pairZ are the pair densities of the focal type with
% ordinary cooperators and with zealots. qC, qZ and qD are the
% probabilities that a neighbour is an ordinary cooperator, a zealot or
% an ordinary defector. If the focal type is (numerically) absent, a
% harmless default is returned; the corresponding terms are multiplied
% by that density.

if total > tiny
    qC = max(pairC/total,0);
    qZ = max(pairZ/total,0);
    s  = qC + qZ;
    if s > 1
        qC = qC/s;
        qZ = qZ/s;
        qD = 0;
    else
        qD = 1 - s;
    end
else
    qC = 0;
    qZ = 0;
    qD = 1;
end

end

%% ==========================================================
function Y = sanitize(Y,p)
% Project the state onto the admissible region:
%   0 <= x <= 1,  0 <= pZC <= min(pC, z - pZZ),  0 <= pCC <= pC - pZC.

Y   = Y(:);
x   = min(max(Y(1),0),1);
pC  = p.u*x;
pZC = min(max(Y(3),0),min(pC,max(p.z-p.pZZ,0)));
pCC = min(max(Y(2),0),max(pC-pZC,0));
Y   = [x; pCC; pZC];

end

%% ==========================================================
function Y0 = init_state(x0,p)
% Initial state with ordinary cooperation x0 and the pair densities of a
% randomly mixed network: pCC = (u x0)^2 and pZC = z u x0, u = 1 - z.

pC0 = p.u*x0;
Y0  = sanitize([x0; pC0^2; p.z*pC0],p);

end

%% ==========================================================
function p = set_z(p,z)
% Zealot fraction with random zealot placement (pZZ = z^2).

p.z   = z;
p.u   = 1 - z;
p.pZZ = z^2;

end

%% ==========================================================
function p = set_payoffs(p,R,S,T,P)
% Store the payoffs R, S, T and P in the parameter structure p.

p.R = R;
p.S = S;
p.T = T;
p.P = P;

end

%% ==========================================================
function coeff = make_coeff(k)
% Multinomial coefficients k!/(kZ! kC! kD!).

coeff = zeros(k+1,k+1);
for kZ = 0:k
    for kC = 0:(k-kZ)
        coeff(kZ+1,kC+1) = nchoosek(k,kZ)*nchoosek(k-kZ,kC);
    end
end

end

%% ==========================================================
function [Y,method,res] = solve_ss(beta,Y0,p)
% Integrate in passes of length p.pass_time until a steady state is found.
% Inputs: beta (selection intensity), Y0 (initial state), p (parameters).
% Outputs: Y (final state), method (acceptance code below) and res, the
% max-norm of the unscaled right-hand side at Y.
%   method = 1 : unscaled residual below p.rhs_tol
%   method = 2 : change over one pass below p.tail_tol (if p.allow_tail)
%   method = 3 : absorbing boundary accepted (mu = 0 only)
%   method = 4 : NOT converged after p.max_pass passes

opts   = odeset('RelTol',p.reltol,'AbsTol',p.abstol,'MaxStep',p.max_step);
Y      = sanitize(Y0,p);
method = 4;
res    = Inf;

for pass = 1:p.max_pass
    Yold = Y;
    [~,Ysol] = ode15s(@(t,y) rhs(y,beta,p),[0 p.pass_time],Y,opts);
    Y = sanitize(Ysol(end,:).',p);

    Fraw = rhs_raw(Y,beta,p);
    res  = norm(Fraw,inf);

    if res < p.rhs_tol
        method = 1;
        return;
    end
    if p.allow_tail && pass >= p.min_pass && max(abs(Y-Yold)) < p.tail_tol
        method = 2;
        return;
    end
    if boundary_accepted(Y,Fraw,p)
        method = 3;
        return;
    end
end

end

%% ==========================================================
function ok = boundary_accepted(Y,Fraw,p)
% For mu = 0 the states x = 0 (only if z = 0) and x = 1 are absorbing.
% Accept them when the trajectory sits on the boundary and is not
% being pushed back into the interior.

x  = Y(1);
xd = Fraw(1);
near_zero = (x < p.bound_tol)     && (xd <=  p.rhs_tol);
near_one  = (x > 1 - p.bound_tol) && (xd >= -p.rhs_tol);
ok = (p.mu == 0) && (near_zero || near_one);

end

%% ==========================================================
function safe_save(fig,filename)
% Save a figure as a 300-dpi PNG; fall back to saveas if print fails.

if ishandle(fig)
    drawnow;
    pause(0.05);
    try
        set(fig,'PaperPositionMode','auto');
        print(fig,filename,'-dpng','-r300');
    catch
        try
            saveas(fig,filename);
        catch
            warning('Could not save figure: %s',filename);
        end
    end
end

end

%% ==========================================================
function s = method_summary(methods)
% Text summary of steady-state acceptance codes.
% (Not called in this script; kept so that the shared functions are
% the same in every script.)

m = methods(:);
s = sprintf(['  residual = %d, tail = %d, boundary = %d, NOT converged = %d'], ...
    sum(m==1),sum(m==2),sum(m==3),sum(m==4));

end
