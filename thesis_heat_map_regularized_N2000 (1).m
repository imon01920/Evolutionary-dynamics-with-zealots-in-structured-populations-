function [all_maps_pC,all_maps_x] = thesis_heat_map_regularized_N2000(beta,mu)
% THESIS_HEAT_MAP_REGULARIZED_N2000  Heat maps of total cooperation
% rho* = z + (1-z) x* and ordinary cooperation x* on the (S,T) plane
% (R = 1, P = 0) for several zealot fractions (MATLAB R2016a).
%
% The steady state of the pair-approximation equations is computed on a
% 41 x 41 grid of S in [-1, 1] and T in [0, 3] for six zealot fractions,
% starting from x0 = 0.40 with random-mixing pair densities. The lines
% T = 1 and S = 0 divide each panel into the trivial (Harmony) game TR,
% the Chicken game CG, the Stag Hunt game SH and the Prisoner's Dilemma
% PD. The default run gives Fig. 4(b) (ordinary cooperation x*) and
% Fig. C1 (total cooperation rho*) of the paper.
%
% Inputs:
%   beta - selection intensity (default 1)
%   mu   - mutation rate of ordinary players (default 0.10)
%
% Outputs:
%   all_maps_pC - rho*(S,T,z), array of size 41 x 41 x 6
%   all_maps_x  - x*(S,T,z),   array of size 41 x 41 x 6
%
% Usage:
%   thesis_heat_map_regularized_N2000            % beta = 1, mu = 0.1
%   thesis_heat_map_regularized_N2000(1,0.2)     % beta = 1, mu = 0.2
%
% Output files (<tag> is, e.g., beta_1_mu_0p10):
%   heat_map_total_pC_<tag>_N2000.png    rho* (colour-bar label p_C^*)
%   heat_map_ordinary_x_<tag>_N2000.png  x*
%   heat_map_<tag>_N2000_results.mat     both maps, acceptance codes and
%                                        parameters
% Points that do not converge are left transparent (white) and counted.
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

if nargin < 1 || isempty(beta)
    beta = 1.0;
end
if nargin < 2 || isempty(mu)
    mu = 0.10;
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
par.pass_time = 400*par.N;
par.max_step  = 10*par.N;
par.max_pass  = 120;
par.min_pass  = 3;
par.coeff     = make_coeff(par.k);

% Payoff plane: R = 1 and P = 0 are fixed, S and T vary on a 41 x 41
% grid; six zealot fractions
R = 1;
P = 0;
z_vals = [0 0.05 0.10 0.20 0.30 0.40];
S_vals = linspace(-1,1,41);
T_vals = linspace(0,3,41);

nS = length(S_vals);
nT = length(T_vals);
nZ = length(z_vals);

% Results indexed (S, T, z); method_all keeps the acceptance codes
all_maps_pC = NaN(nS,nT,nZ);
all_maps_x  = NaN(nS,nT,nZ);
method_all  = zeros(nS,nT,nZ);

%% -------------------- main computation --------------------
% Steady state for every zealot fraction and every grid point (S,T),
% starting from x0 with random-mixing pair densities
for zi = 1:nZ
    pz = set_z(par,z_vals(zi));
    fprintf('Running z = %.2f, beta = %g, mu = %g, N = %d\n', ...
        pz.z,beta,par.mu,par.N);
    for si = 1:nS
        for ti = 1:nT
            p = set_payoffs(pz,R,S_vals(si),T_vals(ti),P);
            [Y,method] = solve_ss(beta,init_state(p.x0,p),p);
            method_all(si,ti,zi) = method;
            if method ~= 4
                x_star = min(max(Y(1),0),1);
                all_maps_x(si,ti,zi)  = x_star;
                all_maps_pC(si,ti,zi) = p.z + p.u*x_star;
            end
        end
    end
end

%% -------------------- plots --------------------
% fig1: total cooperation rho* (colour-bar label p_C^*);
% fig2: ordinary cooperation x*
fig1 = plot_heatmaps(all_maps_pC,T_vals,S_vals,z_vals,'p_C^*');
fig2 = plot_heatmaps(all_maps_x ,T_vals,S_vals,z_vals,'x^*');

% File-name tag, e.g. beta_1_mu_0p10
tag = sprintf('beta_%s_mu_%s',strrep(sprintf('%g',beta),'.','p'), ...
                              strrep(sprintf('%.2f',mu),'.','p'));
safe_save(fig1,['heat_map_total_pC_' tag '_N2000.png']);
safe_save(fig2,['heat_map_ordinary_x_' tag '_N2000.png']);

%% -------------------- output --------------------
fprintf('\nSteady-state acceptance:\n%s\n',method_summary(method_all));
if any(method_all(:) == 4)
    fprintf('Non-converged points are shown as white. Increase max_pass/pass_time.\n');
end

save(['heat_map_' tag '_N2000_results.mat'],'all_maps_pC','all_maps_x', ...
    'method_all','S_vals','T_vals','z_vals','beta','par');

disp('Heat maps complete.');

end

%% ==========================================================
function fig = plot_heatmaps(M,T_vals,S_vals,z_vals,colorbar_label)
% Draw a 2 x 3 grid of heat maps, one for each zealot fraction, with a
% common colour bar from 0 to 1.
%   M              - values of size numel(S_vals) x numel(T_vals) x
%                    numel(z_vals)
%   T_vals, S_vals - payoff grids (T on the horizontal axis)
%   z_vals         - zealot fractions (panel titles)
%   colorbar_label - text of the colour-bar label (TeX)

fig = figure('Color','w','Position',[40 50 1300 720]);

% Panel size, panel positions and colour-bar position (normalized units)
plot_w = 0.205;
plot_h = 0.315;
x_pos  = [0.070 0.335 0.600];
y_pos  = [0.560 0.125];
cb_pos = [0.845 0.125 0.018 0.750];

last_ax = [];

for zi = 1:length(z_vals)
    row = ceil(zi/3);
    col = zi - 3*(row-1);
    ax = axes('Parent',fig,'Position',[x_pos(col) y_pos(row) plot_w plot_h]);
    last_ax = ax;

    Mz = M(:,:,zi);
    axes(ax);                                       %#ok<LAXES> make current
    hImg = imagesc(T_vals,S_vals,Mz);
    set(hImg,'AlphaData',double(~isnan(Mz)));       % NaN -> transparent
    set(ax,'YDir','normal','Layer','top','Color','w');
    caxis(ax,[0 1]);
    hold(ax,'on');

    % Lines S = 0 and T = 1 separate the four game classes
    line([min(T_vals) max(T_vals)],[0 0],'Parent',ax,'Color','k','LineWidth',1.3);
    line([1 1],[min(S_vals) max(S_vals)],'Parent',ax,'Color','k','LineWidth',1.3);

    % Game-class labels (TR = trivial, or Harmony, game)
    text(0.28, 0.68,'TR','Parent',ax,'FontWeight','bold','FontSize',18,'Color','k');
    text(1.55, 0.68,'CG','Parent',ax,'FontWeight','bold','FontSize',18,'Color','k');
    text(0.28,-0.72,'SH','Parent',ax,'FontWeight','bold','FontSize',18,'Color','k');
    text(1.55,-0.72,'PD','Parent',ax,'FontWeight','bold','FontSize',18,'Color','k');

    if row == 2
        xlabel(ax,'T','FontWeight','bold');
    end
    if col == 1
        ylabel(ax,'S','FontWeight','bold');
    end
    title(ax,sprintf('z = %.2f',z_vals(zi)),'FontWeight','bold');

    xlim(ax,[min(T_vals) max(T_vals)]);
    ylim(ax,[min(S_vals) max(S_vals)]);
    set(ax,'FontSize',18);
    hold(ax,'off');
end

colormap(fig,parula(256));
hcb = colorbar('peer',last_ax);
set(hcb,'Position',cb_pos);
ylabel(hcb,colorbar_label,'FontWeight','bold','Interpreter','tex');

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
