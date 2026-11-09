function thesis_branch_bistability_merging_N2000()
% THESIS_BRANCH_BISTABILITY_MERGING_N2000  Critical-mass (branch-merging)
% test for PD, SH and CG with mu = 0 (MATLAB R2016a).
%
% For every (game, beta, z) the steady state is computed from a low
% (x0 = 1e-4) and a high (x0 = 1 - 1e-4) initial condition. A critical
% mass is reported at z_c if the branches are separated (gap > gapTol) on
% at least minBiPoints positive z values, and afterwards merge
% (gap <= mergeTol) with both branches at full cooperation (x >= xTarget).
% z_c on the grid is then refined by bisection. The paper reports the
% grid values of z_c; the refined values are only printed.
%
% Classes printed for every (game, beta):
%   CME            - critical mass: separated branches that merge at full
%                    cooperation (the criterion above)
%   smoothFull     - no critical mass, but both branches reach full
%                    cooperation for some z (smooth response)
%   bistableNoFull - separated branches that do not merge at full
%                    cooperation in the scanned range
%   noCME          - no critical mass and no full cooperation
%
% Usage:
%   thesis_branch_bistability_merging_N2000
%
% Output files:
%   SH_CME_N2000.png                      branch curves of the Stag Hunt
%                                         game, beta = 0.01, 2, 7 (Fig. 7(a))
%   PD_CME_N2000.png, CG_CME_N2000.png    the same for the Prisoner's
%                                         Dilemma and the Chicken game
%                                         (Fig. C5)
%   branch_gap_N2000.png                  branch gap |x_high - x_low| for
%                                         all beta and z (Fig. 7(b))
%   branch_bistability_N2000_results.mat  branches, gaps, z_c values,
%                                         classes and parameters
%
% Expected (SH, R = 1, S = -0.5, T = 0.5, P = 0):
%   beta = 2 : grid z_c = 0.065, refined z_c ~ 0.0630
%   beta = 7 : grid z_c = 0.130, refined z_c ~ 0.1261
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

clc; close all;

%% -------------------- parameters --------------------
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

% Selection intensities, the betas shown in the branch figures, and the
% grid of zealot fractions
betaList  = [0.01 0.05 0.1 0.5 1 2 3 5 7 10];
showBetas = [0.01 2 7];
zList     = 0:0.005:0.50;

% Branch test: initial x of the low and high branches, separation and
% merging tolerances for the gap, level of full cooperation, minimum
% number of separated positive z values, and bisection steps for z_c
xLow0       = 1e-4;
xHigh0      = 1 - 1e-4;
gapTol      = 1e-3;
mergeTol    = 1e-3;
xTarget     = 1 - 1e-8;
minBiPoints = 2;
nRefine     = 12;          % bisection steps for z_c (0 = no refinement)

% Payoffs (R,S,T,P) of the three games
games = struct('short',{'PD','SH','CG'}, ...
               'R',{1,1,1}, 'S',{-0.5,-0.5,0.5}, ...
               'T',{1.5,0.5,1.5}, 'P',{0,0,0});

nG = length(games);
nB = length(betaList);
nZ = length(zList);

% Results indexed (game, beta, z); methodAll(g,b,branch,z) keeps the
% acceptance codes (branch 1 = low, 2 = high)
xLowAll   = NaN(nG,nB,nZ);
xHighAll  = NaN(nG,nB,nZ);
gapAll    = NaN(nG,nB,nZ);
methodAll = zeros(nG,nB,2,nZ);
zCrit     = NaN(nG,nB);
zCritRef  = NaN(nG,nB);
hasCME    = false(nG,nB);
hasFull   = false(nG,nB);
hasBi     = false(nG,nB);

fprintf('\nBranch-bistability test, N = %d, mu = %g\n',p.N,p.mu);
fprintf('%6s %8s %10s %10s %12s %14s\n','Game','beta','maxGap','zCrit','zCrit(ref)','class');

%% -------------------- main computation --------------------
% For every game and beta: both branches over the z grid, the
% classification and, for a critical mass, bisection between the last
% separated grid point and z_c
for g = 1:nG
    pg = set_payoffs(p,games(g).R,games(g).S,games(g).T,games(g).P);

    for b = 1:nB
        beta = betaList(b);
        [xLow,xHigh,methods] = sweep_game(beta,pg,zList,xLow0,xHigh0);
        methodAll(g,b,:,:) = reshape(methods,[1 1 2 nZ]);

        gap  = abs(xHigh - xLow);
        xMin = min(xLow,xHigh);

        xLowAll(g,b,:)  = xLow;
        xHighAll(g,b,:) = xHigh;
        gapAll(g,b,:)   = gap;

        [zc,isCME,isFull,isBi,iMerge] = classify(gap,xMin,zList, ...
            gapTol,mergeTol,xTarget,minBiPoints);
        zCrit(g,b)  = zc;
        hasCME(g,b) = isCME;
        hasFull(g,b)= isFull;
        hasBi(g,b)  = isBi;

        if isCME && nRefine > 0 && iMerge > 1
            zCritRef(g,b) = refine_zc(beta,pg,zList(iMerge-1),zList(iMerge), ...
                xLow0,xTarget,nRefine);
        end

        if isCME
            cls = 'CME';
        elseif isFull
            cls = 'smoothFull';
        elseif isBi
            cls = 'bistableNoFull';
        else
            cls = 'noCME';
        end

        fprintf('%6s %8.3g %10.3e %10s %12s %14s\n', ...
            games(g).short,beta,max(gap),fmt_z(zc,3),fmt_z(zCritRef(g,b),4),cls);
        if any(methods(:) == 4)
            fprintf('       WARNING: %d non-converged points for this beta.\n', ...
                sum(methods(:) == 4));
        end
    end
end

fprintf('\nSteady-state acceptance:\n%s\n',method_summary(methodAll));

%% -------------------- branch figures --------------------
for g = 1:nG
    plot_branch_figure(g,games,betaList,showBetas,zList,xLowAll,xHighAll,zCrit);
end

%% -------------------- branch-gap heat map --------------------
% One row per beta value (the beta grid is not uniform in log scale, so
% the rows are indexed 1..nB and labelled with the beta values).
fig = figure('Color','w','Position',[80 120 1450 430]);
yLabs = cell(1,nB);
for i = 1:nB
    yLabs{i} = num2str(betaList(i));
end
cmax = max(gapAll(:));
if ~(cmax > 0)
    cmax = 1;
end

last_ax = [];
for g = 1:nG
    ax = subplot(1,3,g);
    last_ax = ax;
    imagesc(zList,1:nB,squeeze(gapAll(g,:,:)));
    set(ax,'YDir','normal','YTick',1:nB,'YTickLabel',yLabs, ...
        'FontName','Times New Roman','FontSize',18,'LineWidth',1.0);
    colormap(ax,parula(256));
    caxis(ax,[0 cmax]);
    xlabel(ax,'Zealot fraction (z)','FontWeight','bold','FontSize',18);
    if g == 1
        ylabel(ax,'Selection intensity (\beta)','FontWeight','bold', ...
            'Interpreter','tex','FontSize',18);
    end
    title(ax,[games(g).short ': branch gap'],'FontWeight','bold');
    xlim(ax,[min(zList) max(zList)]);
    ylim(ax,[0.5 nB+0.5]);
end

hcb = colorbar('peer',last_ax);
set(hcb,'Position',[0.930 0.160 0.015 0.700]);
ylabel(hcb,'Branch gap','FontWeight','bold','FontSize',16);

safe_save(fig,'branch_gap_N2000.png');

save('branch_bistability_N2000_results.mat','xLowAll','xHighAll','gapAll', ...
    'methodAll','zCrit','zCritRef','hasCME','hasFull','hasBi', ...
    'betaList','zList','games','p');

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
% returned.

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

%% ==========================================================
function plot_branch_figure(g,games,betaList,showBetas,zList,xLowAll,xHighAll,zCrit)
% Branch curves of game g for the betas in showBetas: steady states from
% the low (blue) and high (red) initial conditions, and a vertical line
% at z_c when a critical mass is found. The high branch is drawn shifted
% by dzShow in z, so that it stays visible where the two branches
% coincide.

colLow  = [0.000 0.250 0.650];
colHigh = [0.850 0.180 0.050];
colCrit = [0.000 0.520 0.250];
dzShow  = 0.0015;

fig = figure('Color','w','Position',[80 100 1650 500]);
ax_pos = [0.060 0.150 0.235 0.720; ...
          0.335 0.150 0.235 0.720; ...
          0.610 0.150 0.235 0.720];
branch_ylim = [-0.05 1.02];

hHigh1 = [];
hLow1  = [];
hCrit1 = [];

for j = 1:length(showBetas)
    [~,b] = min(abs(betaList-showBetas(j)));
    xLow  = squeeze(xLowAll(g,b,:)).';
    xHigh = squeeze(xHighAll(g,b,:)).';

    ax = axes('Parent',fig,'Position',ax_pos(j,:));
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    hLow  = plot(ax,zList,xLow,'-o','Color',colLow,'LineWidth',2.1, ...
        'MarkerSize',4.5,'MarkerFaceColor',colLow);
    hHigh = plot(ax,zList+dzShow,xHigh,'--s','Color',colHigh,'LineWidth',2.0, ...
        'MarkerSize',5.5,'MarkerFaceColor','w');
    if isempty(hHigh1)
        hHigh1 = hHigh;
        hLow1  = hLow;
    end
    if ~isnan(zCrit(g,b))
        hc = plot(ax,[zCrit(g,b) zCrit(g,b)],branch_ylim,'-.', ...
            'Color',colCrit,'LineWidth',1.8);
        if isempty(hCrit1)
            hCrit1 = hc;
        end
    end

    title(ax,[games(g).short '(\beta = ' num2str(betaList(b)) ')'], ...
        'FontWeight','bold','Interpreter','tex','FontSize',16);
    xlabel(ax,'Zealot fraction (z)','FontWeight','bold','FontSize',20);
    if j == 1
        ylabel(ax,'Ordinary cooperation (x^*)','FontWeight','bold','FontSize',20);
    end
    xlim(ax,[min(zList) max(zList)+dzShow]);
    ylim(ax,branch_ylim);
    set(ax,'FontName','Times New Roman','FontSize',18);
end

if isempty(hCrit1)
    lgd = legend([hHigh1 hLow1],{'High initial','Low initial'},'FontSize',14);
else
    lgd = legend([hHigh1 hLow1 hCrit1], ...
        {'High initial','Low initial','Transition line'},'FontSize',14);
end
set(lgd,'Position',[0.865 0.385 0.115 0.230]);

safe_save(fig,[games(g).short '_CME_N2000.png']);

end

%% ==========================================================
function s = fmt_z(z,nd)
% Format z with nd decimals for the printed table ('---' for NaN).

if isnan(z)
    s = '---';
else
    s = sprintf(['%.' num2str(nd) 'f'],z);
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

m = methods(:);
s = sprintf(['  residual = %d, tail = %d, boundary = %d, NOT converged = %d'], ...
    sum(m==1),sum(m==2),sum(m==3),sum(m==4));

end
