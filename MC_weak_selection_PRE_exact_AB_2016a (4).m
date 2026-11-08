function MC_weak_selection_PRE_exact_AB_2016a(N,mu)
% MC_WEAK_SELECTION_PRE_EXACT_AB_2016A  Explicit-network Monte Carlo test
% of the weak-selection condition A_mu (R-T) + B_mu (S-P) > 0
% (MATLAB R2016a).
%
% PART I  : pair-approximation coefficients A_mu, B_mu (neutral steady
%           state, Jacobian J_mu, structural vectors U_mu, V_mu).
% PART II : paired Monte Carlo chains (beta > 0 and beta = 0, same graph,
%           same initial state, common random numbers) on random
%           k-regular graphs; weak-selection slope fitted across beta and
%           compared with the analytical zero crossing.
%
% For every repetition, beta and swept payoff value, the chain with
% selection and the neutral chain give the difference of the time-averaged
% ordinary cooperation, dx = x_beta - x_0. The slope is dx/beta for each
% beta and, combining the three betas, the least-squares fit dx = s*beta
% through the origin. Means and 95% confidence intervals are taken over
% the repetitions, and the zero crossing of the mean fitted slope is
% compared with the analytical crossing. The five runs below give Fig. 6,
% Figs. C3-C4 and Tables 1-2 of the paper.
%
% Inputs:
%   N  - number of nodes of the random regular graph (default 1000)
%   mu - mutation rate of ordinary players (default 1e-3)
%
% Usage (the five runs used in the paper):
%   MC_weak_selection_PRE_exact_AB_2016a            % N = 1000, mu = 1e-3
%   MC_weak_selection_PRE_exact_AB_2016a(2000)      % N = 2000, mu = 1e-3
%   MC_weak_selection_PRE_exact_AB_2016a(5000)      % N = 5000, mu = 1e-3
%   MC_weak_selection_PRE_exact_AB_2016a(2000,1e-4) % N = 2000, mu = 1e-4
%   MC_weak_selection_PRE_exact_AB_2016a(2000,1e-2) % N = 2000, mu = 1e-2
%
% Output files (<tag> is, e.g., N2000_mu1em03 for N = 2000, mu = 1e-3):
%   MC_weak_fitted_<tag>.png       fitted slope with its 95% band
%                                  (Fig. 6 and Fig. C4)
%   MC_weak_beta_slopes_<tag>.png  slope for each beta (Fig. C3)
%   MC_weak_summary_<tag>.csv      slopes and confidence intervals
%   MC_weak_results_<tag>.mat      all results
%   MC_weak_checkpoint_<tag>.mat   checkpoint, saved after every repetition
%
% All output files carry N and mu in their names, so runs never overwrite
% each other. An interrupted run resumes from its checkpoint file. When
% all repetitions are finished, running the same command again only
% redraws the figures; delete the checkpoint file to simulate again.
% A run can take several hours.
%
% Expected analytical values (k = 6, z = 0.1, mu = 1e-3, R = 1.2,
% S = -0.4, P = 0): x0_mu = 0.97674825..., A_mu = 1.73928e-01,
% B_mu = 3.65802e-02, predicted crossing T = 1.1158725...
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

if nargin < 1 || isempty(N)
    N = 1000;
end
if nargin < 2 || isempty(mu)
    mu = 1e-3;
end
clc; close all;

%% ============================================================
% PART I. PAIR-APPROXIMATION PARAMETERS
%% ============================================================
% Degree k, mutation rate mu, network size N and zealot fraction z = 0.10
par.k  = 6;
par.mu = mu;
par.N  = N;
par    = set_z(par,0.10);

% Solver settings for the neutral steady state (see solve_ss); fdY is
% the relative step of the finite-difference Jacobian (num_jac).
par.rhs_tol    = 1e-10;     % strict target for the neutral residual
par.tail_tol   = 0;
par.bound_tol  = 1e-8;
par.allow_tail = false;
par.min_pass   = 1;
par.reltol     = 1e-9;
par.abstol     = 1e-11;
par.fdY        = 1e-6;
par.tiny       = 1e-14;
par.max_pass   = 50;
par.pass_time  = 50*par.N;
par.max_step   = 5*par.N;
par.coeff      = make_coeff(par.k);

par.neutral_residual_accept = 1e-8;   % hard upper limit for the residual

% Payoffs; the swept entry (T or S) replaces the value given here
base.R = 1.20;
base.S = -0.40;
base.T = 1.60;            % only used if S is swept
base.P = 0.00;

% Selection intensities, number of swept payoff values and half-width
% of the sweep around the predicted crossing
beta_small = [0.0025 0.005 0.010];
nSweep     = 41;
sweepSpan  = 1.0;

%% ============================================================
% PART I-A. NEUTRAL REGULARIZED STEADY STATE
%% ============================================================
par0 = set_payoffs(par,0,0,0,0);
[Y0,method0,F0] = solve_ss(0,init_state(0.50,par0),par0);

if F0 > par.neutral_residual_accept
    error(['Neutral PA steady state not accurate enough: residual %.4e > %.4e. ' ...
           'Increase par.max_pass or par.pass_time.'],F0,par.neutral_residual_accept);
elseif method0 ~= 1
    warning('Neutral residual %.4e is above %.1e but below %.1e. Proceeding.', ...
        F0,par.rhs_tol,par.neutral_residual_accept);
end
if Y0(1) <= 0 || Y0(1) >= 1
    error('Neutral steady state is not interior (x0 = %.3e).',Y0(1));
end
x0_mu = Y0(1);

%% ============================================================
% PART I-B. JACOBIAN, STRUCTURAL VECTORS, A_mu, B_mu
%% ============================================================
J     = num_jac(Y0,par0);
condJ = cond(J);
[U,V] = structural_vectors(Y0,par0);
Amu   = -[1 0 0]*(J\U);
Bmu   = -[1 0 0]*(J\V);

if condJ > 1e10
    warning('Neutral Jacobian is ill-conditioned: cond(J) = %.4e.',condJ);
end

%% ============================================================
% PART I-C. PAYOFF SWEEP AND ANALYTICAL CROSSING
%% ============================================================
if abs(Bmu) >= abs(Amu)
    if abs(Bmu) < 1e-14
        error('B_mu is too close to zero for an S-crossing calculation.');
    end
    sweepName   = 'S';
    cross_pred  = base.P - (Amu/Bmu)*(base.R-base.T);
    sweepVals   = linspace(cross_pred-sweepSpan,cross_pred+sweepSpan,nSweep);
    theoryCurve = Amu*(base.R-base.T) + Bmu*(sweepVals-base.P);
    xlab        = 'Swept payoff entry S';
else
    if abs(Amu) < 1e-14
        error('A_mu is too close to zero for a T-crossing calculation.');
    end
    sweepName   = 'T';
    cross_pred  = base.R + (Bmu/Amu)*(base.S-base.P);
    sweepVals   = linspace(cross_pred-sweepSpan,cross_pred+sweepSpan,nSweep);
    theoryCurve = Amu*(base.R-sweepVals) + Bmu*(base.S-base.P);
    xlab        = 'Swept payoff entry T';
end
theory_cross = zero_crossing_2016a(sweepVals,theoryCurve);

fprintf('\n============================================================\n');
fprintf('ANALYTICAL STRUCTURAL COEFFICIENTS (pair approximation)\n');
fprintf('============================================================\n');
fprintf('Neutral raw residual             = %.4e\n',F0);
fprintf('cond(J_mu)                       = %.4e\n',condJ);
fprintf('x0_mu                            = %.10f\n',x0_mu);
fprintf('A_mu                             = %.12e\n',Amu);
fprintf('B_mu                             = %.12e\n',Bmu);
fprintf('sigma_mu = A_mu/B_mu             = %.8f\n',Amu/Bmu);
fprintf('Sweep entry                      = %s\n',sweepName);
fprintf('Predicted analytical crossing    = %.10f\n',cross_pred);
fprintf('Theory curve zero crossing       = %.10f\n',theory_cross);
fprintf('============================================================\n\n');

%% ============================================================
% PART II. MONTE CARLO SETTINGS
%% ============================================================
mc.N  = par.N;
mc.k  = par.k;
mc.z  = par.z;
mc.mu = par.mu;
mc.xInit = x0_mu;          % start MC at the neutral PA level

% nRep repetitions, each with its own graph and zealot placement.
% The first burnSweeps sweeps are discarded; x is then recorded every
% sampleEvery sweeps for measureSweeps sweeps (1 sweep = N update
% attempts). swapFactor*N*k/2 edge swaps randomize each graph.
% tailDriftWarn is the warning level for the difference between the
% first and second halves of the measurement window.
nRep          = 30;
burnSweeps    = 2500;
measureSweeps = 2500;
maxSweeps     = burnSweeps + measureSweeps;
sampleEvery   = 10;
swapFactor    = 15;
tailDriftWarn = 0.003;

nB = length(beta_small);
nT = length(sweepVals);

% File-name tag, e.g. N2000_mu1em03 for N = 2000, mu = 1e-3
tag = sprintf('N%d_mu%s',mc.N,strrep(sprintf('%.0e',mc.mu),'-','m'));
checkpointFile = ['MC_weak_checkpoint_' tag '.mat'];

%% ============================================================
% PART II-A. STORAGE (or resume from checkpoint)
%% ============================================================
% Arrays indexed (repetition, beta, swept value):
%   xBetaMeanAll, xZeroMeanAll - time-averaged x of the chain with
%                                selection and of the neutral chain
%   deltaXAll, slopeAll        - their difference, and difference/beta
%   tailDriftBetaAll/ZeroAll   - drift between the two halves of the
%                                measurement window
% fitSlopeRep(rep,ti) is the slope fitted through the origin over the
% betas, and crossRep(rep) is its zero crossing.
xBetaMeanAll     = NaN(nRep,nB,nT);
xZeroMeanAll     = NaN(nRep,nB,nT);
deltaXAll        = NaN(nRep,nB,nT);
slopeAll         = NaN(nRep,nB,nT);
fitSlopeRep      = NaN(nRep,nT);
tailDriftBetaAll = NaN(nRep,nB,nT);
tailDriftZeroAll = NaN(nRep,nB,nT);
crossRep         = NaN(nRep,1);
elapsedByRep     = NaN(nRep,1);
lastRepDone      = 0;

% Resume only from a checkpoint made with the same settings
if exist(checkpointFile,'file') == 2
    C = load(checkpointFile);
    sameSetup = isfield(C,'lastRepDone') && isfield(C,'sweepVals') && ...
        isequal(size(C.xBetaMeanAll),[nRep nB nT]) && ...
        max(abs(C.sweepVals - sweepVals)) < 1e-12 && ...
        isequal(C.beta_small,beta_small) && C.maxSweeps == maxSweeps;
    if sameSetup
        xBetaMeanAll     = C.xBetaMeanAll;
        xZeroMeanAll     = C.xZeroMeanAll;
        deltaXAll        = C.deltaXAll;
        slopeAll         = C.slopeAll;
        fitSlopeRep      = C.fitSlopeRep;
        tailDriftBetaAll = C.tailDriftBetaAll;
        tailDriftZeroAll = C.tailDriftZeroAll;
        crossRep         = C.crossRep;
        elapsedByRep     = C.elapsedByRep;
        lastRepDone      = C.lastRepDone;
        fprintf('Resuming from checkpoint: %d / %d repetitions done.\n',lastRepDone,nRep);
    else
        warning('Checkpoint %s has different settings; starting fresh.',checkpointFile);
    end
end

fprintf('\n============================================================\n');
fprintf('MONTE CARLO WEAK-SELECTION VALIDATION\n');
fprintf('============================================================\n');
fprintf('N = %d, k = %d, z = %.3f, mu = %.4g\n',mc.N,mc.k,mc.z,mc.mu);
fprintf('Initial MC x              = %.10f\n',mc.xInit);
fprintf('beta values               = %s\n',sprintf('%.4g ',beta_small));
fprintf('Sweep %s from %.5f to %.5f (%d points)\n',sweepName, ...
    min(sweepVals),max(sweepVals),nT);
fprintf('Repetitions = %d, burn-in = %d, measurement = %d, sample every %d sweeps\n', ...
    nRep,burnSweeps,measureSweeps,sampleEvery);
fprintf('============================================================\n\n');

globalTic = tic;

%% ============================================================
% PART II-B. MONTE CARLO MAIN LOOP
%% ============================================================
for rep = (lastRepDone+1):nRep
    repTic = tic;
    fprintf('\n--- Repetition %d / %d ---\n',rep,nRep);

    % Reproducible, resume-safe graph and placement for this repetition.
    rng(500000 + rep);
    neighbors    = random_regular_neighbors_switching_2016a(mc.N,mc.k,swapFactor);
    zealotOrder  = randperm(mc.N);
    stateInitial = initialize_state_zero_crossing_2016a(mc.N,mc.z,mc.xInit,zealotOrder);

    for b = 1:nB
        beta = beta_small(b);
        fprintf('   beta = %.4g\n',beta);
        seed = 1000000 + 10000*rep + 100*b;

        % Neutral (beta = 0) chain. With common random numbers it is the
        % same at every sweep point, so it is simulated once per (rep, b)
        % with the same random stream that every beta > 0 chain uses.
        [probZeroD2C,probZeroC2D] = precompute_switch_probs_2016a(0,mc,0,0,0,0);
        rng(seed);
        [xZeroMean,tailZero] = run_MC_chain_2016a(stateInitial,neighbors, ...
            probZeroD2C,probZeroC2D,mc,maxSweeps,burnSweeps,sampleEvery);

        for ti = 1:nT
            sweepNow = sweepVals(ti);
            if mod(ti,10) == 1 || ti == nT
                fprintf('      %s = %.5f   (%d/%d)\n',sweepName,sweepNow,ti,nT);
            end

            if strcmp(sweepName,'S')
                Rnow = base.R; Snow = sweepNow; Tnow = base.T; Pnow = base.P;
            else
                Rnow = base.R; Snow = base.S; Tnow = sweepNow; Pnow = base.P;
            end

            [probBetaD2C,probBetaC2D] = precompute_switch_probs_2016a(beta,mc,Rnow,Snow,Tnow,Pnow);

            rng(seed);                     % common random numbers
            [xBetaMean,tailBeta] = run_MC_chain_2016a(stateInitial,neighbors, ...
                probBetaD2C,probBetaC2D,mc,maxSweeps,burnSweeps,sampleEvery);

            xBetaMeanAll(rep,b,ti)     = xBetaMean;
            xZeroMeanAll(rep,b,ti)     = xZeroMean;
            deltaXAll(rep,b,ti)        = xBetaMean - xZeroMean;
            slopeAll(rep,b,ti)         = (xBetaMean - xZeroMean)/beta;
            tailDriftBetaAll(rep,b,ti) = tailBeta;
            tailDriftZeroAll(rep,b,ti) = tailZero;
        end
    end

    % Least-squares fit through the origin: deltaX(beta) ~ beta*slope.
    for ti = 1:nT
        dvec = reshape(deltaXAll(rep,:,ti),[nB 1]);
        fitSlopeRep(rep,ti) = sum(beta_small(:).*dvec)/sum(beta_small(:).^2);
    end
    crossRep(rep)     = zero_crossing_2016a(sweepVals,fitSlopeRep(rep,:));
    elapsedByRep(rep) = toc(repTic);
    lastRepDone       = rep;

    fprintf('Repetition %d: %.2f min, replicate crossing = %.10f\n', ...
        rep,elapsedByRep(rep)/60,crossRep(rep));

    save(checkpointFile,'par','base','Y0','x0_mu','J','condJ','U','V','Amu','Bmu', ...
        'sweepName','cross_pred','theory_cross','sweepVals','theoryCurve','mc', ...
        'beta_small','nRep','burnSweeps','measureSweeps','maxSweeps','sampleEvery', ...
        'swapFactor','xBetaMeanAll','xZeroMeanAll','deltaXAll','slopeAll', ...
        'fitSlopeRep','tailDriftBetaAll','tailDriftZeroAll','crossRep', ...
        'elapsedByRep','lastRepDone');
end

totalRuntime = toc(globalTic);

%% ============================================================
% PART II-C. STATISTICS
%% ============================================================
% Means over the repetitions and 95% confidence half-widths
% (1.96 standard errors)
slopeMean = reshape(mean(slopeAll,1),[nB nT]);
slopeSE   = reshape(std(slopeAll,0,1),[nB nT])./sqrt(nRep);
slopeCI95 = 1.96*slopeSE;

fitSlopeMean = mean(fitSlopeRep,1);
fitSlopeSE   = std(fitSlopeRep,0,1)./sqrt(nRep);
fitSlopeCI95 = 1.96*fitSlopeSE;

% Crossing of the mean fitted curve, and the mean and 95% interval of
% the crossings of the single repetitions
mcCrossFromMeanCurve = zero_crossing_2016a(sweepVals,fitSlopeMean);

validCross = crossRep(~isnan(crossRep));
if ~isempty(validCross)
    mcCrossMean = mean(validCross);
    mcCrossSE   = std(validCross)/sqrt(length(validCross));
    mcCrossCI95 = 1.96*mcCrossSE;
else
    mcCrossMean = NaN;
    mcCrossSE   = NaN;
    mcCrossCI95 = NaN;
end

% Convergence check: drift between the two halves of the measurement
% window, for the chains with selection and the neutral chains
maxTailDriftBeta  = max(tailDriftBetaAll(:));
maxTailDriftZero  = max(tailDriftZeroAll(:));
meanTailDriftBeta = mean(tailDriftBetaAll(~isnan(tailDriftBetaAll)));
meanTailDriftZero = mean(tailDriftZeroAll(~isnan(tailDriftZeroAll)));

fprintf('\n============================================================\n');
fprintf('FINAL SUMMARY (N = %d, mu = %.4g)\n',mc.N,mc.mu);
fprintf('============================================================\n');
fprintf('x0_mu                                  = %.10f\n',x0_mu);
fprintf('A_mu                                   = %.12e\n',Amu);
fprintf('B_mu                                   = %.12e\n',Bmu);
fprintf('cond(J_mu)                             = %.4e\n',condJ);
fprintf('Analytical predicted crossing          = %.10f\n',cross_pred);
fprintf('MC crossing from mean fitted curve     = %.10f\n',mcCrossFromMeanCurve);
fprintf('Relative deviation                     = %+.2f %%\n', ...
    100*(mcCrossFromMeanCurve-cross_pred)/cross_pred);
fprintf('Mean replicate MC crossing             = %.10f\n',mcCrossMean);
fprintf('95%% CI half-width for MC crossing      = %.10f\n',mcCrossCI95);
if ~isnan(mcCrossMean)
    if abs(mcCrossMean - cross_pred) <= mcCrossCI95
        fprintf('Theory lies INSIDE the 95%% CI of the replicate crossings.\n');
    else
        fprintf('Theory lies OUTSIDE the 95%% CI of the replicate crossings.\n');
    end
end
fprintf('Valid replicate crossings              = %d / %d\n',length(validCross),nRep);
fprintf('Mean / max tail drift (beta chains)    = %.4e / %.4e\n',meanTailDriftBeta,maxTailDriftBeta);
fprintf('Mean / max tail drift (neutral chains) = %.4e / %.4e\n',meanTailDriftZero,maxTailDriftZero);
if maxTailDriftBeta > tailDriftWarn || maxTailDriftZero > tailDriftWarn
    fprintf('WARNING: visible tail drift. Consider longer burn-in/measurement.\n');
else
    fprintf('Tail-drift diagnostic within tolerance.\n');
end
fprintf('Runtime of this session                = %.3f hours\n',totalRuntime/3600);
fprintf('============================================================\n\n');

%% ============================================================
% FIGURE 1. FITTED MC SLOPE WITH 95% BAND
%% ============================================================
fig1 = figure('Color','w','Position',[90 90 1080 640]);
ax1  = axes('Parent',fig1);
hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
xMinPlot = min(sweepVals);
xMaxPlot = max(sweepVals);

plot_CI_band_2016a(ax1,sweepVals,fitSlopeMean,fitSlopeCI95,[0.90 0.80 0.92]);
hTheory = plot(ax1,sweepVals,theoryCurve,'k-','LineWidth',2.6);
hMC     = plot(ax1,sweepVals,fitSlopeMean,'mo-','LineWidth',1.8, ...
    'MarkerSize',5,'MarkerFaceColor','m');

xlim(ax1,[xMinPlot xMaxPlot]);
yl = ylim(ax1);
hZero        = plot(ax1,[xMinPlot xMaxPlot],[0 0],'k:','LineWidth',1.4);
hTheoryCross = plot(ax1,[cross_pred cross_pred],yl,'--','Color',[0.35 0.35 0.35],'LineWidth',1.6);

hList = [hTheory hMC hZero hTheoryCross];
lList = {'Theoretical graph','Monte Carlo graph','Zero line in x-axis', ...
         'Theoretical zero crossing line'};
if ~isnan(mcCrossFromMeanCurve)
    hMCCross = plot(ax1,[mcCrossFromMeanCurve mcCrossFromMeanCurve],yl,'-.', ...
        'Color',[0.00 0.50 0.20],'LineWidth',1.8);
    hList = [hList hMCCross];
    lList{end+1} = 'Monte Carlo zero crossing line';
end
xlim(ax1,[xMinPlot xMaxPlot]);
ylim(ax1,yl);
xlabel(ax1,xlab,'FontWeight','bold','FontSize',18);
ylabel(ax1,'Weak selection slope','FontWeight','bold','FontSize',18);
title(ax1,sprintf('N = %d, \\mu = %g',mc.N,mc.mu),'FontWeight','bold');
hLegend1 = legend(hList,lList,'Location','eastoutside');
set(hLegend1,'FontSize',16,'Box','on');
set(ax1,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);
safe_save_png_2016a(fig1,['MC_weak_fitted_' tag '.png']);

%% ============================================================
% FIGURE 2. BETA-SPECIFIC MC SLOPES
%% ============================================================
fig2 = figure('Color','w','Position',[100 100 1080 640]);
ax2  = axes('Parent',fig2);
hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');

hTheory2 = plot(ax2,sweepVals,theoryCurve,'k-','LineWidth',2.6);
styles = {'b--o','r-.s','g:^','c--d','k-.v'};
hBeta  = gobjects(1,nB);
lBeta  = cell(1,nB);
for b = 1:nB
    hBeta(b) = plot(ax2,sweepVals,slopeMean(b,:),styles{1+mod(b-1,numel(styles))}, ...
        'LineWidth',1.6,'MarkerSize',4.5);
    lBeta{b} = ['Monte Carlo graph, \beta = ' num2str(beta_small(b))];
end
xlim(ax2,[xMinPlot xMaxPlot]);
yl2 = ylim(ax2);
hZero2        = plot(ax2,[xMinPlot xMaxPlot],[0 0],'k:','LineWidth',1.4);
hTheoryCross2 = plot(ax2,[cross_pred cross_pred],yl2,'--','Color',[0.35 0.35 0.35],'LineWidth',1.6);
xlim(ax2,[xMinPlot xMaxPlot]);
ylim(ax2,yl2);
xlabel(ax2,xlab,'FontWeight','bold','FontSize',18);
ylabel(ax2,'Weak selection slope','FontWeight','bold','FontSize',18);
title(ax2,sprintf('N = %d, \\mu = %g',mc.N,mc.mu),'FontWeight','bold');
hLegend2 = legend([hTheory2 hBeta hZero2 hTheoryCross2], ...
    [{'Theoretical graph'} lBeta {'Zero line in x-axis','Theoretical zero crossing line'}], ...
    'Location','eastoutside');
set(hLegend2,'FontSize',16,'Box','on');
set(ax2,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);
safe_save_png_2016a(fig2,['MC_weak_beta_slopes_' tag '.png']);

%% ============================================================
% CSV EXPORT
%% ============================================================
csvFile = ['MC_weak_summary_' tag '.csv'];
fid = fopen(csvFile,'w');
if fid < 0
    warning('Could not open %s for writing.',csvFile);
else
    header = 'sweepValue,theorySlope,fitSlopeMean,fitSlopeCI95';
    for b = 1:nB
        header = [header sprintf(',slopeBeta%dMean,slopeBeta%dCI95',b,b)]; %#ok<AGROW>
    end
    fprintf(fid,'%s\n',header);
    for ti = 1:nT
        fprintf(fid,'%.12g,%.12g,%.12g,%.12g',sweepVals(ti),theoryCurve(ti), ...
            fitSlopeMean(ti),fitSlopeCI95(ti));
        for b = 1:nB
            fprintf(fid,',%.12g,%.12g',slopeMean(b,ti),slopeCI95(b,ti));
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
end

%% ============================================================
% FINAL MAT FILE
%% ============================================================
resultsFile = ['MC_weak_results_' tag '.mat'];
save(resultsFile,'par','base','Y0','x0_mu','J','condJ','U','V','Amu','Bmu', ...
    'sweepName','cross_pred','theory_cross','sweepVals','theoryCurve','mc', ...
    'beta_small','nRep','burnSweeps','measureSweeps','maxSweeps','sampleEvery', ...
    'swapFactor','xBetaMeanAll','xZeroMeanAll','deltaXAll','slopeAll', ...
    'slopeMean','slopeSE','slopeCI95','fitSlopeRep','fitSlopeMean','fitSlopeSE', ...
    'fitSlopeCI95','tailDriftBetaAll','tailDriftZeroAll','crossRep', ...
    'mcCrossFromMeanCurve','mcCrossMean','mcCrossSE','mcCrossCI95', ...
    'elapsedByRep','totalRuntime');

fprintf('Saved: MC_weak_fitted_%s.png, MC_weak_beta_slopes_%s.png,\n',tag,tag);
fprintf('       %s, %s, %s\n',csvFile,resultsFile,checkpointFile);
fprintf('Finished.\n');

end

%% ============================================================
% ONE MONTE CARLO CHAIN
%% ============================================================
function [xMean,tailDrift] = run_MC_chain_2016a(stateInitial,neighbors, ...
    probD2C,probC2D,p,maxSweeps,burnSweeps,sampleEvery)
% State codes: 0 = ordinary defector, 1 = ordinary cooperator, 2 = zealot.
% Every elementary step draws randi(N) and rand (also when a zealot is
% chosen), so two chains started from the same seed use identical random
% numbers step by step (common random numbers).
% coopNbr(i) is the number of cooperative neighbours (zealots and
% ordinary cooperators) of node i; it is updated whenever a neighbour
% switches. Outputs: xMean is the time average of x over the
% measurement window, and tailDrift is the absolute difference between
% the averages over its second and first halves.

N = p.N;
k = p.k;

state         = stateInitial;
ordinaryCount = sum(stateInitial ~= 2);
ordinaryCoop  = sum(state == 1);
coopNbr       = initial_coop_neighbor_counts_2016a(state,neighbors,N,k);

maxSamples    = floor((maxSweeps-burnSweeps)/sampleEvery) + 2;
samples       = zeros(maxSamples,1);
sampleCounter = 0;

for sweep = 1:maxSweeps
    for attempt = 1:N
        i = randi(N);
        u = rand;
        if state(i) == 2                     % zealots never update
            continue;
        end
        nC = coopNbr(i);
        if state(i) == 0
            if u < probD2C(nC+1)
                state(i) = 1;
                ordinaryCoop = ordinaryCoop + 1;
                nb = neighbors(i,:);
                coopNbr(nb) = coopNbr(nb) + 1;
            end
        else
            if u < probC2D(nC+1)
                state(i) = 0;
                ordinaryCoop = ordinaryCoop - 1;
                nb = neighbors(i,:);
                coopNbr(nb) = coopNbr(nb) - 1;
            end
        end
    end

    if sweep > burnSweeps && mod(sweep,sampleEvery) == 0
        sampleCounter = sampleCounter + 1;
        samples(sampleCounter) = ordinaryCoop/ordinaryCount;
    end
end

if sampleCounter > 0
    xMean     = mean(samples(1:sampleCounter));
    halfPoint = floor(sampleCounter/2);
    if halfPoint >= 2
        tailDrift = abs(mean(samples(halfPoint+1:sampleCounter)) - mean(samples(1:halfPoint)));
    else
        tailDrift = NaN;
    end
else
    xMean     = ordinaryCoop/ordinaryCount;
    tailDrift = NaN;
end

end

%% ============================================================
% MONTE CARLO SWITCHING PROBABILITIES (same kernels as the PA model)
%% ============================================================
function [probD2C,probC2D] = precompute_switch_probs_2016a(beta,p,R,S,T,P)
% Switching probabilities of an ordinary player with nCoop = 0,...,k
% cooperative neighbours (zealots count as cooperators):
%   probD2C(nCoop+1) = mu + (1-mu) A F( beta*Delta)    (D -> C)
%   probC2D(nCoop+1) = mu + (1-mu) B F(-beta*Delta)    (C -> D)
% with A = nCoop/k, B = 1 - A, Delta = A(R-T) + B(S-P) and F the Fermi
% function.

k = p.k;
probD2C = zeros(k+1,1);
probC2D = zeros(k+1,1);
for nCoop = 0:k
    A = nCoop/k;
    B = 1 - A;
    Delta = A*(R-T) + B*(S-P);
    probD2C(nCoop+1) = p.mu + (1-p.mu)*A*fermi( beta*Delta);
    probC2D(nCoop+1) = p.mu + (1-p.mu)*B*fermi(-beta*Delta);
end

end

%% ============================================================
% INITIAL MONTE CARLO STATE
%% ============================================================
function state = initialize_state_zero_crossing_2016a(N,z,xInit,zealotOrder)
% Initial state: the first round(z*N) nodes of zealotOrder are zealots
% (state 2), and a random fraction xInit of the ordinary nodes are
% cooperators (state 1). At least one ordinary cooperator and one
% ordinary defector are kept.

state = zeros(N,1);
nZ = round(z*N);
state(zealotOrder(1:nZ)) = 2;

ordinaryNodes = find(state ~= 2);
nOrd = length(ordinaryNodes);
nC = round(xInit*nOrd);
nC = min(max(1,nC),nOrd-1);

permOrd = ordinaryNodes(randperm(nOrd));
state(permOrd(1:nC)) = 1;

end

%% ============================================================
% INITIAL COOPERATIVE-NEIGHBOUR COUNTS (zealots count as cooperators)
%% ============================================================
function coopNbr = initial_coop_neighbor_counts_2016a(state,neighbors,N,k)
% Number of cooperative neighbours (zealots and ordinary cooperators)
% of every node.

isCoop  = (state > 0);
coopNbr = zeros(N,1);
for jj = 1:k
    coopNbr = coopNbr + isCoop(neighbors(:,jj));
end

end

%% ============================================================
% RANDOM k-REGULAR GRAPH BY DEGREE-PRESERVING DOUBLE-EDGE SWAPS
%% ============================================================
function neighbors = random_regular_neighbors_switching_2016a(N,k,swapFactor)
% Starts from a ring lattice (k/2 neighbours on each side) and performs
% swapFactor*M successful swaps (a,b),(c,d) -> (a,d),(c,b) or
% (a,c),(b,d), rejecting self-loops and multiple edges.
% M = N*k/2 is the number of edges. Output: neighbors(i,:) lists the k
% neighbours of node i.

if mod(k,2) ~= 0
    error('This graph generator assumes even k.');
end
if k >= N
    error('k must be smaller than N.');
end

halfK = k/2;
M = N*k/2;
edges = zeros(M,2);
idx = 0;
for d = 1:halfK
    for i = 1:N
        j = i + d;
        if j > N
            j = j - N;
        end
        idx = idx + 1;
        edges(idx,:) = [min(i,j) max(i,j)];
    end
end

if N <= 10000
    A = false(N,N);                         % fast dense adjacency
else
    A = logical(sparse(N,N));
end
for e = 1:M
    A(edges(e,1),edges(e,2)) = true;
    A(edges(e,2),edges(e,1)) = true;
end

targetSwaps     = swapFactor*M;
successfulSwaps = 0;
attempts        = 0;
maxAttempts     = 300*targetSwaps;

while successfulSwaps < targetSwaps && attempts < maxAttempts
    attempts = attempts + 1;
    e1 = randi(M);
    e2 = randi(M);
    if e1 == e2
        continue;
    end
    a = edges(e1,1);  b = edges(e1,2);
    if rand < 0.5                           % random orientation of edge 2
        c = edges(e2,1);  d = edges(e2,2);
    else
        c = edges(e2,2);  d = edges(e2,1);
    end

    % new edges (a,d) and (c,b)
    if a == d || c == b
        continue;                           % self-loop
    end
    if A(a,d) || A(c,b)
        continue;                           % multiple edge
    end
    u1 = min(a,d); v1 = max(a,d);
    u2 = min(c,b); v2 = max(c,b);
    if u1 == u2 && v1 == v2
        continue;
    end

    A(a,b) = false;  A(b,a) = false;
    A(c,d) = false;  A(d,c) = false;
    A(u1,v1) = true; A(v1,u1) = true;
    A(u2,v2) = true; A(v2,u2) = true;
    edges(e1,:) = [u1 v1];
    edges(e2,:) = [u2 v2];
    successfulSwaps = successfulSwaps + 1;
end

if successfulSwaps < targetSwaps
    warning('Graph randomization stopped after %d of %d swaps.',successfulSwaps,targetSwaps);
end

neighbors = zeros(N,k);
deg = zeros(N,1);
for e = 1:M
    u = edges(e,1);
    v = edges(e,2);
    deg(u) = deg(u) + 1;  neighbors(u,deg(u)) = v;
    deg(v) = deg(v) + 1;  neighbors(v,deg(v)) = u;
end
if any(deg ~= k)
    error('Graph construction failed: degree mismatch.');
end
for i = 1:N
    if any(neighbors(i,:) == i) || length(unique(neighbors(i,:))) ~= k
        error('Graph construction failed: self-loop or multiple edge at node %d.',i);
    end
end

end

%% ============================================================
% ZERO CROSSING BY LINEAR INTERPOLATION
%% ============================================================
function crossVal = zero_crossing_2016a(x,y)
% First zero crossing of y(x) by linear interpolation (NaN if none).

crossVal = NaN;
for i = 1:(length(x)-1)
    y1 = y(i);
    y2 = y(i+1);
    if isnan(y1) || isnan(y2)
        continue;
    end
    if y1 == 0
        crossVal = x(i);
        return;
    end
    if y1*y2 < 0 || y2 == 0
        crossVal = x(i) - y1*(x(i+1)-x(i))/(y2-y1);
        return;
    end
end

end

%% ============================================================
% 95% CONFIDENCE BAND
%% ============================================================
function plot_CI_band_2016a(ax,x,meanY,ciY,bandColor)
% Shaded band meanY +/- ciY on the axes ax; NaN points are skipped.

x     = x(:).';
meanY = meanY(:).';
ciY   = ciY(:).';
ok    = ~isnan(meanY) & ~isnan(ciY);
x = x(ok); meanY = meanY(ok); ciY = ciY(ok);
if numel(x) < 2
    return;
end
fill([x fliplr(x)],[meanY+ciY fliplr(meanY-ciY)],bandColor, ...
    'EdgeColor','none','Parent',ax);

end

%% ============================================================
% SAFE PNG SAVE
%% ============================================================
function safe_save_png_2016a(fig,filename)
% Save a figure as a 600-dpi PNG; fall back to saveas if print fails.

if ishandle(fig)
    drawnow;
    pause(0.05);
    try
        set(fig,'PaperPositionMode','auto');
        print(fig,filename,'-dpng','-r600');
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
%  WEAK-SELECTION TOOLS
%% ==========================================================
function ws = weak_selection_coefficients(p,x_init)
% Neutral regularized equilibrium Y0 and structural coefficients
%   A_mu = -e1' J^{-1} U,   B_mu = -e1' J^{-1} V.
% x_init is the initial x used to find the neutral state. The output
% structure ws has the fields Y0, x0, res (residual), J, cond (condition
% number of J), A and B.
% (Not called in this script; kept so that the shared functions are
% the same in every script.)

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
% (Not called in this script; kept so that the shared functions are
% the same in every script.)

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
% (Not called in this script; kept so that the shared functions are
% the same in every script.)

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
% (Not called in this script; kept so that the shared functions are
% the same in every script.)

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
