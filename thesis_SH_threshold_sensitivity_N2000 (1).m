function thesis_SH_threshold_sensitivity_N2000()
% THESIS_SH_THRESHOLD_SENSITIVITY_N2000  Critical zealot fraction z_c of
% the Stag Hunt game versus beta, for several T (S fixed) and several S
% (T fixed), with mu = 0 (MATLAB R2016a). The criterion is identical to
% thesis_branch_bistability_merging_N2000.
%
% z_c is the first grid value of z at which the low and high branches
% merge at full cooperation after a bistable interval; it is NaN (and not
% plotted) when there is no critical mass. With crit.nRefine = 0 the grid
% values are used, as in the paper. The result is Fig. 9 of the paper.
%
% Usage:
%   thesis_SH_threshold_sensitivity_N2000
%
% Output files:
%   SH_threshold_sensitivity_S_N2000.png  z_c versus beta, varying S
%                                         (Fig. 9(a))
%   SH_threshold_sensitivity_T_N2000.png  z_c versus beta, varying T
%                                         (Fig. 9(b))
%   SH_threshold_sensitivity_N2000_results.mat
%                                         z_c values, classes, parameters
% Tables of z_c are also printed in the command window.
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

clc; close all;

%% -------------------- model settings --------------------
% Model: degree k, no mutation (mu = 0) and population size N (time
% scale only)
p.k  = 6;
p.mu = 0;
p.N  = 2000;

% Steady-state test (see solve_ss): a state is accepted when the unscaled
% residual max|N dY/dt| is below rhs_tol or, if allow_tail is true, when
% Y changes by less than tail_tol over one pass. bound_tol is used for
% the absorbing states x = 0 and x = 1 (mu = 0 only), tiny is the
% smallest density treated as nonzero, and reltol, abstol are the ode15s
% tolerances.
p.rhs_tol    = 1e-10;
p.tail_tol   = 1e-7;
p.bound_tol  = 1e-8;
p.tiny       = 1e-14;
p.allow_tail = true;
p.reltol     = 1e-9;
p.abstol     = 1e-11;

% Integration in passes of pass_time update steps (scaled with N,
% because time is measured in single update steps), with ode15s steps of
% at most max_step; at most max_pass passes, and at least min_pass passes
% before the tail test is used. coeff holds the multinomial coefficients.
p.pass_time = 350*p.N;
p.max_step  = 5*p.N;
p.max_pass  = 1000;
p.min_pass  = 3;
p.coeff     = make_coeff(p.k);

% Fixed payoffs
R = 1.0;
P = 0.0;

S_fixed = -0.50;                 % family A: vary T at fixed S
T_list  = [0.20 0.50 0.80];
T_fixed = 0.50;                  % family B: vary S at fixed T
S_list  = [-0.80 -0.50 -0.20];

% Selection intensities and grid of zealot fractions
beta_vec = [0.01 0.05 0.1 0.5 1 2 3 5 7 10];
z_vec    = 0:0.005:0.50;

% Branch-test criterion, as in thesis_branch_bistability_merging_N2000
crit.gapTol      = 1e-3;
crit.mergeTol    = 1e-3;
crit.xTarget     = 1 - 1e-8;
crit.minBiPoints = 2;
crit.xLow0       = 1e-4;
crit.xHigh0      = 1 - 1e-4;
crit.nRefine     = 0;            % set e.g. 12 to refine z_c by bisection

% One colour per curve
colors = [0.0000 0.2500 0.6500; ...
          0.8500 0.1800 0.0500; ...
          0.0000 0.5200 0.2500];

count = zeros(1,4);              % [residual, tail, boundary, NOT converged]

%% -------------------- varying T --------------------
zCrit_T = NaN(length(T_list),length(beta_vec));
class_T = cell(length(T_list),length(beta_vec));
fprintf('\nSH threshold sensitivity, N = %d\n',p.N);
fprintf('Varying T at fixed S = %.2f\n',S_fixed);
for i = 1:length(T_list)
    pg = set_payoffs(p,R,S_fixed,T_list(i),P);
    [zCrit_T(i,:),class_T(i,:),cNow] = compute_threshold_curve(pg,beta_vec,z_vec,crit);
    count = count + cNow;
    fprintf('  T = %.2f done\n',T_list(i));
end

%% -------------------- varying S --------------------
zCrit_S = NaN(length(S_list),length(beta_vec));
class_S = cell(length(S_list),length(beta_vec));
fprintf('Varying S at fixed T = %.2f\n',T_fixed);
for i = 1:length(S_list)
    pg = set_payoffs(p,R,S_list(i),T_fixed,P);
    [zCrit_S(i,:),class_S(i,:),cNow] = compute_threshold_curve(pg,beta_vec,z_vec,crit);
    count = count + cNow;
    fprintf('  S = %.2f done\n',S_list(i));
end

%% -------------------- tables --------------------
print_table('Critical zealot fraction z_c, varying T',beta_vec,T_list,'T',zCrit_T);
print_table('Critical zealot fraction z_c, varying S',beta_vec,S_list,'S',zCrit_S);

fprintf('\nSteady-state acceptance\n');
fprintf('  residual = %d, tail = %d, boundary = %d, NOT converged = %d\n', ...
    count(1),count(2),count(3),count(4));

%% -------------------- figures --------------------
fig1 = plot_curves(beta_vec,zCrit_T,T_list,'T',colors,'o-', ...
    sprintf('SH critical threshold: varying T  (S = %.2f)',S_fixed),max(z_vec));
safe_save(fig1,'SH_threshold_sensitivity_T_N2000.png');

fig2 = plot_curves(beta_vec,zCrit_S,S_list,'S',colors,'s-', ...
    sprintf('SH critical threshold: varying S  (T = %.2f)',T_fixed),max(z_vec));
safe_save(fig2,'SH_threshold_sensitivity_S_N2000.png');

save('SH_threshold_sensitivity_N2000_results.mat','zCrit_T','zCrit_S', ...
    'class_T','class_S','beta_vec','z_vec','T_list','S_list','S_fixed','T_fixed','p','crit');

end

%% ============================================================
function [zCritVec,classVec,count] = compute_threshold_curve(p,beta_vec,z_vec,crit)
% z_c for every beta in beta_vec, for one payoff set p. Outputs:
% zCritVec (NaN where there is no critical mass), classVec (class names
% as in thesis_branch_bistability_merging_N2000) and count, the numbers
% of steady states accepted by each test of solve_ss.

nB = length(beta_vec);
zCritVec = NaN(1,nB);
classVec = cell(1,nB);
count    = zeros(1,4);

for b = 1:nB
    beta = beta_vec(b);
    [xLow,xHigh,methods] = sweep_game(beta,p,z_vec,crit.xLow0,crit.xHigh0);
    for m = 1:4
        count(m) = count(m) + sum(methods(:) == m);
    end

    gap  = abs(xHigh - xLow);
    xMin = min(xLow,xHigh);
    [zc,isCME,isFull,isBi,iMerge] = classify(gap,xMin,z_vec, ...
        crit.gapTol,crit.mergeTol,crit.xTarget,crit.minBiPoints);

    if isCME && crit.nRefine > 0 && iMerge > 1
        zc = refine_zc(beta,p,z_vec(iMerge-1),z_vec(iMerge), ...
            crit.xLow0,crit.xTarget,crit.nRefine);
    end
    zCritVec(b) = zc;

    if isCME
        classVec{b} = 'CME';
    elseif isFull
        classVec{b} = 'smoothFull';
    elseif isBi
        classVec{b} = 'bistableNoFull';
    else
        classVec{b} = 'noCME';
    end
end

end

%% ==========================================================
function [zc,isCME,isFull,isBi,iMerge] = classify(gap,xMin,zList, ...
    gapTol,mergeTol,xTarget,minBiPoints)
% Branch-merging classification.
%   gap  - |x_high - x_low| on the z grid
%   xMin - min(x_low, x_high) on the z grid
% Outputs: zc (critical zealot fraction on the grid, NaN if none), isCME,
% isFull (both branches at full cooperation for some z), isBi (at least
% minBiPoints separated positive z values) and iMerge (grid index of zc).

nZ     = length(zList);
zc     = NaN;
iMerge = 0;
isCME  = false;

biIdx  = find((gap > gapTol) & (zList > 0));
isBi   = length(biIdx) >= minBiPoints;
isFull = any(xMin >= xTarget);

if isBi
    startIdx = biIdx(end) + 1;
    if startIdx <= nZ
        idx = find((gap(startIdx:end) <= mergeTol) & ...
                   (xMin(startIdx:end) >= xTarget),1,'first');
        if ~isempty(idx)
            iMerge = startIdx + idx - 1;
            zc     = zList(iMerge);
            isCME  = true;
        end
    end
end

end

%% ==========================================================
function zc = refine_zc(beta,p,zlo,zhi,xLow0,xTarget,nIter)
% Bisection on z: at zlo the low-initial trajectory stays low, at zhi it
% reaches full cooperation. After nIter halvings the upper end zhi is
% returned. Used only when crit.nRefine > 0.

for it = 1:nIter
    zm = 0.5*(zlo+zhi);
    pz = set_z(p,zm);
    Y  = solve_ss(beta,init_state(xLow0,pz),pz);
    if Y(1) >= xTarget
        zhi = zm;
    else
        zlo = zm;
    end
end
zc = zhi;

end

%% ==========================================================
function [xLow,xHigh,methods] = sweep_game(beta,p,zList,xLow0,xHigh0)
% Each point starts from fresh random-mixing initial conditions.
% Outputs: xLow, xHigh (steady-state x from the low and high initial
% conditions; NaN if not converged) and methods(branch,i), the
% acceptance codes of solve_ss (row 1 low, row 2 high).

nZ      = length(zList);
xLow    = NaN(1,nZ);
xHigh   = NaN(1,nZ);
methods = zeros(2,nZ);

for i = 1:nZ
    pz = set_z(p,zList(i));
    [Ylow,methods(1,i)] = solve_ss(beta,init_state(xLow0,pz),pz);
    if methods(1,i) ~= 4
        xLow(i) = Ylow(1);
    end
    [Yhigh,methods(2,i)] = solve_ss(beta,init_state(xHigh0,pz),pz);
    if methods(2,i) ~= 4
        xHigh(i) = Yhigh(1);
    end
end

end

%% ============================================================
function fig = plot_curves(beta_vec,zCrit,vals,name,colors,style,ttl,zmax)
% Plot z_c versus beta (logarithmic axis) for every value in vals. NaN
% values (no critical mass) are not drawn, and only curves with at least
% one detected critical mass get a legend entry.

fig = figure('Color','w','Position',[120 120 780 510]);
ax  = axes('Parent',fig);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');
set(ax,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0,'XScale','log');

h   = gobjects(0);
leg = {};
for i = 1:length(vals)
    idx = ~isnan(zCrit(i,:));
    if any(idx)
        h(end+1) = plot(ax,beta_vec(idx),zCrit(i,idx),style, ...
            'LineWidth',2.2,'MarkerSize',7,'Color',colors(i,:), ...
            'MarkerFaceColor','w','MarkerEdgeColor',colors(i,:)); %#ok<AGROW>
        leg{end+1} = sprintf('%s = %.2f',name,vals(i));           %#ok<AGROW>
    end
end

xlabel(ax,'Selection intensity (\beta)','FontWeight','bold','Interpreter','tex','FontSize',18);
ylabel(ax,'Critical zealot fraction (z_c)','FontWeight','bold','Interpreter','tex','FontSize',18);
title(ax,ttl,'FontWeight','bold','Interpreter','tex','FontSize',18);
if ~isempty(h)
    legend(h,leg,'Location','eastoutside');
end
xlim(ax,[min(beta_vec) max(beta_vec)]);
ylim(ax,[0 zmax]);
set(ax,'YTick',0:0.05:zmax);

end

%% ============================================================
function print_table(titleText,beta_vec,paramList,paramName,zCrit)
% Print z_c for every beta (rows) and parameter value (columns).

fprintf('\n============================================================\n');
fprintf('%s\n',titleText);
fprintf('--- means no branch-merging critical mass was found.\n');
fprintf('============================================================\n');
fprintf('%8s','beta');
for i = 1:length(paramList)
    fprintf('%12s',sprintf('%s=%.2f',paramName,paramList(i)));
end
fprintf('\n');
for b = 1:length(beta_vec)
    fprintf('%8.3g',beta_vec(b));
    for i = 1:length(paramList)
        if isnan(zCrit(i,b))
            fprintf('%12s','---');
        else
            fprintf('%12.4f',zCrit(i,b));
        end
    end
    fprintf('\n');
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
