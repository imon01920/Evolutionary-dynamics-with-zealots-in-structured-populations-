function thesis_zero_crossing_validation_N2000()
% THESIS_ZERO_CROSSING_VALIDATION_N2000  Validates the weak-selection
% condition A_mu (R-T) + B_mu (S-P) > 0 against steady states of the full
% pair-approximation system (MATLAB R2016a).
%
% Steps:
%   1. Find the neutral (beta = 0) steady state Y0 = (x0_mu, pCC, pZC),
%      the Jacobian J_mu there and the structural vectors U_mu, V_mu, and
%      compute A_mu = -e1' J^{-1} U_mu and B_mu = -e1' J^{-1} V_mu.
%   2. Sweep one payoff entry (T for the default parameters) over 41
%      values around the predicted crossing T = R + (B_mu/A_mu)(S-P).
%   3. At each value, solve for the steady state x*(beta) for
%      beta = 0.0025, 0.005 and 0.01, and form the slopes
%      (x*(beta) - x0_mu)/beta and their fit through the origin.
%   4. Compare the zero crossing of the fitted slope with the prediction.
% The result is Fig. 5 of the paper.
%
% Usage:
%   thesis_zero_crossing_validation_N2000
%
% Output files:
%   zero_crossing_validation_N2000.png  figure
%   zero_crossing_validation_N2000.mat  coefficients, slopes and crossings
%
% Expected output for the default parameters (k = 6, z = 0.1, mu = 1e-3,
% R = 1.2, S = -0.4, P = 0):
%   x0_mu = 0.97674825..., A_mu = 1.73928e-01, B_mu = 3.65802e-02,
%   predicted crossing T = 1.11587259, numerical crossing ~ 1.11586.
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

clc; close all;

%% -------------------- model parameters --------------------
% Degree k, mutation rate mu, population size N (time scale only) and
% zealot fraction z = 0.10
par.k  = 6;
par.mu = 1e-3;
par.N  = 2000;
par    = set_z(par,0.10);

% Solver settings (see solve_ss). Only the residual test is used
% (allow_tail = false), because the weak-selection slopes need accurate
% steady states. fdY is the relative step of the finite-difference
% Jacobian (num_jac).
par.rhs_tol    = 1e-10;   % convergence on the unscaled residual
par.tail_tol   = 0;       % not used (allow_tail = false)
par.bound_tol  = 1e-8;
par.allow_tail = false;   % weak-selection slopes need true steady states
par.min_pass   = 1;
par.reltol     = 1e-9;
par.abstol     = 1e-11;
par.fdY        = 1e-6;
par.tiny       = 1e-14;
par.max_pass   = 50;
par.pass_time  = 50*par.N;
par.max_step   = 5*par.N;
par.coeff      = make_coeff(par.k);

% Payoffs; the swept entry (T or S) replaces the value given here
base.R = 1.20;
base.S = -0.40;
base.T = 1.60;            % only used if S is swept
base.P = 0.00;

% Number of swept values, half-width of the sweep around the predicted
% crossing, and the small selection intensities
nSweep     = 41;
sweepSpan  = 1.0;
beta_small = [0.0025 0.005 0.01];

%% -------------------- neutral state and coefficients --------------------
ws  = weak_selection_coefficients(par,0.50);
x0  = ws.x0;
Amu = ws.A;
Bmu = ws.B;

%% -------------------- choose the swept payoff entry --------------------
% Sweep the entry whose coefficient is larger in magnitude (T for the
% default parameters), centred on the predicted crossing. theory is the
% predicted slope A_mu (R-T) + B_mu (S-P) along the sweep.
if abs(Bmu) >= abs(Amu)
    sweepName  = 'S';
    cross_pred = base.P - (Amu/Bmu)*(base.R-base.T);
    sweep  = linspace(cross_pred-sweepSpan,cross_pred+sweepSpan,nSweep);
    theory = Amu*(base.R-base.T) + Bmu*(sweep-base.P);
    xlab   = 'Swept payoff entry S';
else
    sweepName  = 'T';
    cross_pred = base.R + (Bmu/Amu)*(base.S-base.P);
    sweep  = linspace(cross_pred-sweepSpan,cross_pred+sweepSpan,nSweep);
    theory = Amu*(base.R-sweep) + Bmu*(base.S-base.P);
    xlab   = 'Swept payoff entry T';
end

%% -------------------- numerical weak-selection slopes --------------------
% slope(b,i)   : (x*(beta_b) - x0_mu)/beta_b at the i-th swept value
% fit_slope(i) : least-squares slope through the origin over the betas
% Only steady states that pass the residual test (method = 1) are used.
% Each solve starts from the steady state of the previous beta.
nB        = length(beta_small);
slope     = NaN(nB,nSweep);
fit_slope = NaN(1,nSweep);
n_ok      = 0;
max_res   = 0;

for i = 1:nSweep
    if strcmp(sweepName,'S')
        p = set_payoffs(par,base.R,sweep(i),base.T,base.P);
    else
        p = set_payoffs(par,base.R,base.S,sweep(i),base.P);
    end

    Yseed = ws.Y0;
    dx    = NaN(nB,1);
    for b = 1:nB
        [Yss,method,res] = solve_ss(beta_small(b),Yseed,p);
        max_res = max(max_res,res);
        if method == 1
            n_ok = n_ok + 1;
            dx(b) = Yss(1) - x0;
            slope(b,i) = dx(b)/beta_small(b);
            Yseed = Yss;
        else
            fprintf('  NOT converged: %s = %.5f, beta = %g (res %.2e)\n', ...
                sweepName,sweep(i),beta_small(b),res);
        end
    end
    fit_slope(i) = fit_through_origin(beta_small,dx);
end

% Zero crossings of the fitted numerical slope and of the theory line
num_cross = zero_cross(sweep,fit_slope);
the_cross = zero_cross(sweep,theory);

%% -------------------- summary --------------------
fprintf('\nZero-crossing validation, N = %d\n',par.N);
fprintf('k = %d, z = %.2f, mu = %.1e\n',par.k,par.z,par.mu);
fprintf('Neutral raw residual  = %.3e\n',ws.res);
fprintf('cond(J_mu)            = %.4e\n',ws.cond);
fprintf('x0_mu                 = %.10f\n',x0);
fprintf('A_mu = %.10e, B_mu = %.10e, sigma_mu = %.6f\n',Amu,Bmu,Amu/Bmu);
fprintf('Swept entry           = %s\n',sweepName);
fprintf('Predicted crossing    = %.10f\n',cross_pred);
fprintf('Theory crossing       = %.10f\n',the_cross);
fprintf('Numerical crossing    = %.10f\n',num_cross);
fprintf('Converged slope runs  = %d / %d\n',n_ok,numel(slope));
fprintf('Largest raw residual  = %.3e\n\n',max_res);

%% -------------------- plot --------------------
fig = figure('Color','w','Position',[100 100 1150 560]);
ax = axes('Parent',fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');

xLimits = [min(sweep) max(sweep)];
allY    = [theory(:); slope(:); fit_slope(:)];
allY    = allY(~isnan(allY));
yPad    = 0.08*(max(allY)-min(allY));
yLimits = [min(allY)-yPad, max(allY)+yPad];

% Theory line, slopes for the three betas, fitted slope, zero line and
% predicted crossing
h = gobjects(7,1);
h(1) = plot(ax,sweep,theory,'k-','LineWidth',2.5);
h(2) = plot(ax,sweep,slope(1,:),'b--','LineWidth',2);
h(3) = plot(ax,sweep,slope(2,:),'r-.','LineWidth',2);
h(4) = plot(ax,sweep,slope(3,:),'g:','LineWidth',2);
h(5) = plot(ax,sweep,fit_slope,'m^','MarkerSize',6,'MarkerFaceColor','m');
h(6) = plot(ax,xLimits,[0 0],'k:','LineWidth',1.2);
h(7) = plot(ax,[cross_pred cross_pred],yLimits,'--','Color',[0.35 0.35 0.35],'LineWidth',1.4);

xlim(ax,xLimits);
ylim(ax,yLimits);
xlabel(ax,xlab,'FontWeight','bold');
ylabel(ax,'Weak-selection slope / exact predictor','FontWeight','bold');
legend(h,{'Theory', ...
          sprintf('Slope, \\beta = %g',beta_small(1)), ...
          sprintf('Slope, \\beta = %g',beta_small(2)), ...
          sprintf('Slope, \\beta = %g',beta_small(3)), ...
          'Fitted slope','Zero line','Predicted crossing'}, ...
       'Location','eastoutside','Interpreter','tex');
set(ax,'FontName','Times New Roman','FontSize',18);

safe_save(fig,'zero_crossing_validation_N2000.png');
save('zero_crossing_validation_N2000.mat','ws','sweep','theory','slope', ...
    'fit_slope','beta_small','cross_pred','num_cross','par','base');

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
