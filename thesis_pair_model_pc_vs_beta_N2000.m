function thesis_pair_model_pc_vs_beta_N2000(mu)
% THESIS_PAIR_MODEL_PC_VS_BETA_N2000  Steady-state total cooperation
% rho* = z + (1-z) x* versus the selection intensity beta (MATLAB R2016a).
%
% For the Prisoner's Dilemma (PD), the Stag Hunt game (SH) and the
% Chicken game (CG), the pair-approximation equations are integrated
% until a steady state is reached, for 25 logarithmically spaced values
% of beta in [0.01, 100] and five zealot fractions z. Every run starts
% from x0 = 0.40 with random-mixing pair densities. The three runs below
% give the three rows of Fig. 3 of the paper.
%
% Input:
%   mu - probability of a random strategy switch by an ordinary player
%        (mutation rate); default mu = 0.
%
% Usage:
%   thesis_pair_model_pc_vs_beta_N2000          % mu = 0
%   thesis_pair_model_pc_vs_beta_N2000(0.1)     % mu = 0.1
%   thesis_pair_model_pc_vs_beta_N2000(0.2)     % mu = 0.2
%
% Output files (<mu> is written as, e.g., 0p10 for mu = 0.10):
%   pair_model_pc_vs_beta_mu_<mu>_N2000.png  figure; the y-axis label
%                                            p_C^* is the total
%                                            cooperation rho*
%   pair_model_pc_vs_beta_mu_<mu>_N2000.mat  pc_all(game,z,beta), the
%                                            acceptance codes method_all
%                                            and the parameters
% Points that do not converge are plotted as gaps (NaN) and listed.
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

if nargin < 1 || isempty(mu)
    mu = 0.0;
end
clc; close all;

%% -------------------- parameters --------------------
% Model: population size N (it only sets the time scale), degree k,
% mutation rate mu and initial ordinary cooperation x0
par.N  = 2000;
par.k  = 6;
par.mu = mu;
par.x0 = 0.40;

% Steady-state test (see solve_ss): a state is accepted when the unscaled
% residual max|N dY/dt| is below rhs_tol or, if allow_tail is true, when
% Y changes by less than tail_tol over one pass. bound_tol is used for
% the absorbing states x = 0 and x = 1 (mu = 0 only), tiny is the
% smallest density treated as nonzero, and reltol, abstol are the ode15s
% tolerances.
par.rhs_tol    = 1e-8;
par.tail_tol   = 1e-7;
par.bound_tol  = 1e-8;
par.tiny       = 1e-14;
par.allow_tail = true;
par.reltol     = 1e-8;
par.abstol     = 1e-10;

% Integration in passes of pass_time update steps (scaled with N,
% because time is measured in single update steps), with ode15s steps of
% at most max_step; at most max_pass passes, and at least min_pass passes
% before the tail test is used. coeff holds the multinomial coefficients.
par.pass_time = 500*par.N;
par.max_step  = 10*par.N;
par.max_pass  = 1000;
par.min_pass  = 3;
par.coeff     = make_coeff(par.k);

% Selection intensities (log scale) and zealot fractions
beta_vec = logspace(-2,2,25);
z_vals   = [0 0.05 0.10 0.20 0.30];

% Payoffs (R,S,T,P) of the Prisoner's Dilemma, Stag Hunt and Chicken game
games = struct('name',{'PD','SH','CG'}, ...
               'R',{1.0,1.0,1.0}, ...
               'S',{-0.5,-0.5,0.5}, ...
               'T',{2.0,0.5,2.0}, ...
               'P',{0.0,0.0,0.0});

% One colour per zealot fraction
curve_colors = [0.0000 0.4470 0.7410
                0.8500 0.3250 0.0980
                0.9290 0.6940 0.1250
                0.4940 0.1840 0.5560
                0.4660 0.6740 0.1880];

nG = length(games);
nZ = length(z_vals);
nB = length(beta_vec);

% pc_all(g,i,b): total cooperation rho* for game g, z(i) and beta(b);
% method_all: acceptance code of solve_ss (4 = not converged)
pc_all     = NaN(nG,nZ,nB);
method_all = zeros(nG,nZ,nB);

%% -------------------- main computation --------------------
% Every point starts from x0 with random-mixing pair densities
for g = 1:nG
    pg = set_payoffs(par,games(g).R,games(g).S,games(g).T,games(g).P);
    for zi = 1:nZ
        pz = set_z(pg,z_vals(zi));
        for bi = 1:nB
            beta = beta_vec(bi);
            [Y,method] = solve_ss(beta,init_state(pz.x0,pz),pz);
            method_all(g,zi,bi) = method;
            if method ~= 4
                pc_all(g,zi,bi) = min(max(pz.z + pz.u*Y(1),0),1);
            else
                fprintf('  NOT converged: %s, z = %.2f, beta = %g\n', ...
                    games(g).name,z_vals(zi),beta);
            end
        end
        fprintf('%s, z = %.2f done\n',games(g).name,z_vals(zi));
    end
end

%% -------------------- figure --------------------
% Three panels (PD, SH, CG), one curve for each z; beta on a log scale
fig = figure('Color','w','Position',[100 100 1500 500]);
ax_pos = [0.060 0.150 0.240 0.700;
          0.340 0.150 0.240 0.700;
          0.620 0.150 0.240 0.700];

for g = 1:nG
    ax = axes('Parent',fig,'Position',ax_pos(g,:));
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    for zi = 1:nZ
        plot(ax,beta_vec,squeeze(pc_all(g,zi,:)).','o-', ...
            'LineWidth',2.2,'MarkerSize',6, ...
            'Color',curve_colors(zi,:), ...
            'MarkerFaceColor',curve_colors(zi,:), ...
            'MarkerEdgeColor',curve_colors(zi,:));
    end
    set(ax,'XScale','log');
    xlim(ax,[1e-2 1e2]);
    set(ax,'XTick',[1e-2 1e-1 1 1e1 1e2], ...
           'XTickLabel',{'10^{-2}','10^{-1}','10^0','10^1','10^2'});
    ylim(ax,[-0.05 1.05]);
    set(ax,'YTick',0:0.2:1,'FontSize',18);
    xlabel(ax,'Selection intensity (\beta)','FontWeight','bold', ...
        'FontSize',18,'Interpreter','tex');
    if g == 1
        ylabel(ax,'Fraction of cooperators (p_C^*)', ...
            'FontWeight','bold','FontSize',18,'Interpreter','tex');
    end
    title(ax,games(g).name,'FontWeight','bold','FontSize',17);
end

% Single legend for all panels, drawn in an extra invisible axes
leg_ax = axes('Parent',fig,'Position',[0.885 0.300 0.100 0.360],'Visible','off');
hold(leg_ax,'on');
hleg = gobjects(nZ,1);
legend_text = cell(nZ,1);
for zi = 1:nZ
    hleg(zi) = plot(leg_ax,NaN,NaN,'o-','LineWidth',2.2,'MarkerSize',6, ...
        'Color',curve_colors(zi,:), ...
        'MarkerFaceColor',curve_colors(zi,:), ...
        'MarkerEdgeColor',curve_colors(zi,:));
    legend_text{zi} = sprintf('z = %.2f',z_vals(zi));
end
legend(leg_ax,hleg,legend_text,'Location','northwest','FontSize',16,'Box','on');

%% -------------------- output --------------------
fprintf('\nSteady-state acceptance (mu = %g):\n%s\n',mu,method_summary(method_all));

% File-name tag: mu = 0.10 gives '0p10'
tag = strrep(sprintf('%.2f',mu),'.','p');
safe_save(fig,['pair_model_pc_vs_beta_mu_' tag '_N2000.png']);
save(['pair_model_pc_vs_beta_mu_' tag '_N2000.mat'], ...
    'pc_all','method_all','z_vals','beta_vec','games','par');

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

m = methods(:);
s = sprintf(['  residual = %d, tail = %d, boundary = %d, NOT converged = %d'], ...
    sum(m==1),sum(m==2),sum(m==3),sum(m==4));

end
