function MC_SH_PRE_fine_branch_curves_2016a()
% MC_SH_PRE_FINE_BRANCH_CURVES_2016A  Explicit-network Monte Carlo test of
% the Stag Hunt critical-mass (branch-merging) effect (MATLAB R2016a).
%
% For each repetition a random k-regular graph is built; for every beta and
% zealot fraction z the dynamics is run from a LOW (x0 = 1e-4) and a HIGH
% (x0 = 1 - 1e-4) initial ordinary-cooperation level. The branch gap
% |x_high - x_low| of the replicate means is used to classify the response:
%   TRUE CME          : positive-z bistable window followed by merging at
%                       high cooperation;
%   rapid smooth/full : high cooperation reached without a bistable window.
%
% State labels: 2 = zealot Z, 1 = ordinary cooperator C, 0 = ordinary
% defector D. Switching kernels are identical to the pair-approximation
% model:
%   D -> C : mu + (1-mu) * A * F( beta*(PiC-PiD))
%   C -> D : mu + (1-mu) * B * F(-beta*(PiC-PiD))
% with A (B) the fraction of cooperative (defective) neighbours.
%
% Measured quantity per run: the time average of x over the measurement
% window (sweeps burnSweeps+1 ... maxSweeps). With mu = 0 the state of full
% ordinary cooperation is absorbing; a run that reaches it is recorded as
% x = 1 (and flagged as absorbed). With z = 0 full defection is absorbing
% as well and is recorded as x = 0.
%
% Stochastic classification tolerances (looser than the deterministic
% ones because of finite-N noise): gapTol = mergeTol = 0.05,
% highCoopTol = 0.95, at least 3 positive-z bistable points.
%
% The run is reproducible (fixed seeds per repetition / beta / z / branch)
% and resumes automatically from its checkpoint file after an interruption.
%
% Usage:
%   MC_SH_PRE_fine_branch_curves_2016a
%
% Output files:
%   MC_SH_PRE_fine_branch_curves_2016a.png  branch curves for beta = 0.01,
%                                           2 and 7 (Fig. 8(a))
%   MC_SH_PRE_fine_branch_gap_2016a.png     branch gap (Fig. 8(b))
%   MC_SH_PRE_fine_summary_2016a.csv        means and 95% intervals
%   MC_SH_PRE_fine_results_2016a.mat        all results
%   MC_SH_PRE_fine_checkpoint_2016a.mat     checkpoint, saved after every
%                                           repetition
% The classification (Table 3 of the paper) is printed in the command
% window. When all repetitions are finished, running the script again only
% redraws the figures; delete the checkpoint file to simulate again.
%
% Paper: 'Evolutionary dynamics with zealots in structured populations:
% weak-selection theory and critical-mass transitions'.
% Requirements: MATLAB R2016a or later; no toolboxes are needed.

clc; close all;

%% ============================================================
% MODEL PARAMETERS
%% ============================================================
p.N  = 2000;
p.k  = 6;
p.mu = 0;

% Stag Hunt payoffs used in the thesis
p.R = 1;
p.S = -0.5;
p.T = 0.5;
p.P = 0;

% Selection intensities
betaList = [0.01 2 7];

% Fine grid near the transitions, coarser afterwards
zFine = 0:0.0025:0.16;
zTail = 0.17:0.01:0.24;
zList = unique([zFine zTail]);

% nRep repetitions, each with its own graph and zealot order. A run
% lasts at most maxSweeps sweeps (1 sweep = N update attempts); x is
% recorded every sampleEvery sweeps after the first burnSweeps sweeps.
% swapFactor*N*k/2 edge swaps randomize each graph.
nRep        = 30;
maxSweeps   = 2500;
burnSweeps  = 1700;
sampleEvery = 20;
swapFactor  = 15;

% Initial fraction of ordinary cooperators on the low and high branches
p.xLow0  = 1e-4;
p.xHigh0 = 1 - 1e-4;

% Classification tolerances (stochastic): separation means gap > gapTol
% on at least minBiPoints positive z values; merging means
% gap <= mergeTol with the low branch at x >= highCoopTol. minPositiveZ
% excludes z = 0.
gapTol       = 0.05;
mergeTol     = 0.05;
highCoopTol  = 0.95;
minBiPoints  = 3;
minPositiveZ = 1e-12;

nB = length(betaList);
nZ = length(zList);

checkpointFile = 'MC_SH_PRE_fine_checkpoint_2016a.mat';

%% ============================================================
% STORAGE (or resume from checkpoint)
%% ============================================================
% Results indexed (repetition, beta, z); absorbedLowAll/HighAll mark
% runs that reached full ordinary cooperation
xLowAll         = NaN(nRep,nB,nZ);
xHighAll        = NaN(nRep,nB,nZ);
absorbedLowAll  = false(nRep,nB,nZ);
absorbedHighAll = false(nRep,nB,nZ);
elapsedByRep    = NaN(nRep,1);
lastRepDone     = 0;

if exist(checkpointFile,'file') == 2
    C = load(checkpointFile);
    sameSetup = isfield(C,'lastRepDone') && isfield(C,'zList') && ...
        isequal(size(C.xLowAll),[nRep nB nZ]) && ...
        max(abs(C.zList - zList)) < 1e-12 && isequal(C.betaList,betaList) && ...
        C.maxSweeps == maxSweeps && C.burnSweeps == burnSweeps && C.p.N == p.N;
    if sameSetup
        xLowAll         = C.xLowAll;
        xHighAll        = C.xHighAll;
        absorbedLowAll  = C.absorbedLowAll;
        absorbedHighAll = C.absorbedHighAll;
        elapsedByRep    = C.elapsedByRep;
        lastRepDone     = C.lastRepDone;
        fprintf('Resuming from checkpoint: %d / %d repetitions done.\n',lastRepDone,nRep);
    else
        warning('Checkpoint %s has different settings; starting fresh.',checkpointFile);
    end
end

%% ============================================================
% HEADER
%% ============================================================
fprintf('\n============================================================\n');
fprintf('Stochastic Stag Hunt branch-curve simulation\n');
fprintf('============================================================\n');
fprintf('N = %d, k = %d, mu = %g, (R,S,T,P) = (%g,%g,%g,%g)\n', ...
    p.N,p.k,p.mu,p.R,p.S,p.T,p.P);
fprintf('beta values     = %s\n',sprintf('%.4g ',betaList));
fprintf('z points        = %d (%.4f to %.4f)\n',nZ,min(zList),max(zList));
fprintf('Repetitions     = %d\n',nRep);
fprintf('Sweeps          = %d (burn-in %d, sample every %d)\n', ...
    maxSweeps,burnSweeps,sampleEvery);
fprintf('============================================================\n\n');

globalTic = tic;

%% ============================================================
% MAIN MONTE CARLO LOOP
%% ============================================================
for rep = (lastRepDone+1):nRep
    repTic = tic;
    fprintf('\n--- Repetition %d / %d ---\n',rep,nRep);

    % Graph and nested zealot order for this repetition (reproducible)
    rng(700000 + rep);
    neighbors   = random_regular_neighbors_switching_2016a(p.N,p.k,swapFactor);
    zealotOrder = randperm(p.N);

    for b = 1:nB
        beta = betaList(b);
        fprintf('   beta = %.4g\n',beta);
        [probD2C,probC2D] = precompute_switch_probs_2016a(beta,p);

        for iz = 1:nZ
            z = zList(iz);
            if mod(iz,10) == 1 || iz == nZ
                fprintf('      z = %.4f   (%d/%d)\n',z,iz,nZ);
            end
            seed = 10000000 + 100000*rep + 1000*b + iz;   % < 2^32

            % ---------------- low-initial branch ----------------
            rng(seed);
            stateLow = initialize_branch_state_2016a(p.N,z,zealotOrder,'LOW',p);
            [xLow,absLow] = run_MC_branch_2016a(stateLow,neighbors, ...
                probD2C,probC2D,p,maxSweeps,burnSweeps,sampleEvery,z);
            xLowAll(rep,b,iz)        = xLow;
            absorbedLowAll(rep,b,iz) = absLow;

            % ---------------- high-initial branch ---------------
            rng(seed + 500);
            stateHigh = initialize_branch_state_2016a(p.N,z,zealotOrder,'HIGH',p);
            [xHigh,absHigh] = run_MC_branch_2016a(stateHigh,neighbors, ...
                probD2C,probC2D,p,maxSweeps,burnSweeps,sampleEvery,z);
            xHighAll(rep,b,iz)        = xHigh;
            absorbedHighAll(rep,b,iz) = absHigh;
        end
    end

    elapsedByRep(rep) = toc(repTic);
    lastRepDone = rep;
    fprintf('Repetition %d completed in %.2f minutes.\n',rep,elapsedByRep(rep)/60);

    save(checkpointFile,'p','betaList','zList','nRep','maxSweeps','burnSweeps', ...
        'sampleEvery','swapFactor','xLowAll','xHighAll','absorbedLowAll', ...
        'absorbedHighAll','elapsedByRep','lastRepDone');
end

totalRuntime = toc(globalTic);

%% ============================================================
% STATISTICS
%% ============================================================
% Means over the repetitions and 95% confidence half-widths
% (1.96 standard errors)
xLowMean  = reshape(mean(xLowAll,1),[nB nZ]);
xHighMean = reshape(mean(xHighAll,1),[nB nZ]);
xLowCI95  = 1.96*reshape(std(xLowAll,0,1),[nB nZ])/sqrt(nRep);
xHighCI95 = 1.96*reshape(std(xHighAll,0,1),[nB nZ])/sqrt(nRep);

gapAll  = abs(xHighAll - xLowAll);
gapMean = reshape(mean(gapAll,1),[nB nZ]);
gapCI95 = 1.96*reshape(std(gapAll,0,1),[nB nZ])/sqrt(nRep);

fracAbsLow = reshape(mean(absorbedLowAll,1),[nB nZ]);   % share of runs absorbed

%% ============================================================
% CLASSIFICATION
%% ============================================================
% For each beta: zFirstHigh is the first positive z at which the mean
% low branch is high. A TRUE CME needs a bistable window followed by
% merging at high cooperation (z_c = zCritCME). High cooperation
% without a bistable window is a rapid smooth/full response.
zCritCME               = NaN(1,nB);
zFirstHigh             = NaN(1,nB);
hasPositiveBistability = false(1,nB);
hasTrueCME             = false(1,nB);
hasRapidSmoothFull     = false(1,nB);

for b = 1:nB
    lowMean  = xLowMean(b,:);
    highMean = xHighMean(b,:);
    gap      = abs(highMean - lowMean);

    idxHigh = find((zList > minPositiveZ) & (lowMean >= highCoopTol),1,'first');
    if ~isempty(idxHigh)
        zFirstHigh(b) = zList(idxHigh);
    end

    biIdx = find((zList > minPositiveZ) & (gap > gapTol));
    hasPositiveBistability(b) = length(biIdx) >= minBiPoints;

    if hasPositiveBistability(b)
        lastBiIdx = biIdx(end);
        if lastBiIdx < nZ
            idxMerge = find((gap(lastBiIdx+1:end) <= mergeTol) & ...
                            (lowMean(lastBiIdx+1:end) >= highCoopTol),1,'first');
            if ~isempty(idxMerge)
                zCritCME(b)   = zList(lastBiIdx + idxMerge);
                hasTrueCME(b) = true;
            end
        end
    elseif ~isnan(zFirstHigh(b))
        hasRapidSmoothFull(b) = true;
    end
end

fprintf('\n============================================================\n');
fprintf('Stochastic branch classification\n');
fprintf('============================================================\n');
for b = 1:nB
    if hasTrueCME(b)
        fprintf('beta = %-6g : TRUE CME, z_c ~= %.4f\n',betaList(b),zCritCME(b));
    elseif hasRapidSmoothFull(b)
        fprintf('beta = %-6g : rapid smooth/full response, first high-cooperation z ~= %.4f; NOT a CME\n', ...
            betaList(b),zFirstHigh(b));
    elseif hasPositiveBistability(b)
        fprintf('beta = %-6g : positive-z bistability, but no merge in scanned range\n',betaList(b));
    else
        fprintf('beta = %-6g : no positive-z bistable branch structure\n',betaList(b));
    end
end
fprintf('Runtime of this session = %.3f hours\n',totalRuntime/3600);
fprintf('============================================================\n\n');

%% ============================================================
% FIGURE 1: BRANCH CURVES
%% ============================================================
fig1 = figure('Color','w','Position',[70 100 1550 500]);
lowLineColor  = [0.000 0.300 0.650];  lowBandColor  = [0.780 0.870 0.960];
highLineColor = [0.850 0.250 0.100];  highBandColor = [0.980 0.850 0.780];
critColor     = [0.000 0.500 0.200];

for b = 1:nB
    ax = subplot(1,3,b);
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    plot_CI_band_2016a(ax,zList,xLowMean(b,:),xLowCI95(b,:),lowBandColor);
    plot_CI_band_2016a(ax,zList,xHighMean(b,:),xHighCI95(b,:),highBandColor);

    hLow  = plot(ax,zList,xLowMean(b,:),'-o','Color',lowLineColor,'LineWidth',2.2, ...
        'MarkerSize',4.0,'MarkerFaceColor',lowLineColor);
    hHigh = plot(ax,zList,xHighMean(b,:),'--s','Color',highLineColor,'LineWidth',2.2, ...
        'MarkerSize',4.0,'MarkerFaceColor','w');

    if hasTrueCME(b)
        plot(ax,[zCritCME(b) zCritCME(b)],[0 1.02],'-.','Color',critColor,'LineWidth',1.8);
        title(ax,{['Stag Hunt, \beta = ' num2str(betaList(b))], ...
                  ['Critical mass: z_c \approx ' num2str(zCritCME(b),'%.4f')]}, ...
              'FontWeight','bold','Interpreter','tex');
    elseif hasRapidSmoothFull(b)
        title(ax,{['Stag Hunt, \beta = ' num2str(betaList(b))], ...
                  'Smooth response (no critical mass)'}, ...
              'FontWeight','bold','Interpreter','tex');
    else
        title(ax,['Stag Hunt, \beta = ' num2str(betaList(b))], ...
              'FontWeight','bold','Interpreter','tex');
    end

    xlabel(ax,'Zealot fraction z','FontWeight','bold');
    ylabel(ax,'Ordinary cooperation x^*','FontWeight','bold');
    xlim(ax,[min(zList) max(zList)]);
    ylim(ax,[0 1.02]);
    set(ax,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);
    if b == 1
        legend([hLow hHigh],{'Low initial branch','High initial branch'}, ...
            'Location','southeast');
    end
end
safe_save_png_2016a(fig1,'MC_SH_PRE_fine_branch_curves_2016a.png');

%% ============================================================
% FIGURE 2: BRANCH-GAP HEAT MAP (one row per beta)
%% ============================================================
fig2 = figure('Color','w','Position',[180 140 780 470]);
ax2  = axes('Parent',fig2);
axes(ax2);                                          %#ok<LAXES>
% zList is non-uniform (steps of 0.0025 up to z = 0.16, then steps of
% 0.01). imagesc assumes equally spaced columns, so pcolor is used to
% draw every cell between its true z edges.
zMid   = (zList(1:end-1) + zList(2:end))/2;
zEdges = [max(0,zList(1) - (zList(2)-zList(1))/2), zMid, ...
          zList(end) + (zList(end)-zList(end-1))/2];
yEdges = 0.5:1:(nB+0.5);
Cpad   = NaN(nB+1,nZ+1);           % pcolor uses the last row/column only as edges
Cpad(1:nB,1:nZ) = gapMean;
hPc = pcolor(zEdges,yEdges,Cpad);
set(hPc,'EdgeColor','none');
xlim(ax2,[zEdges(1) zEdges(end)]);
yTickLabels = cell(1,nB);
for b = 1:nB
    yTickLabels{b} = num2str(betaList(b));
end
set(ax2,'YDir','normal','YTick',1:nB,'YTickLabel',yTickLabels, ...
    'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);
ylim(ax2,[0.5 nB+0.5]);
caxis(ax2,[0 1]);
colormap(fig2,parula(256));
hCb = colorbar('peer',ax2);
ylabel(hCb,'Branch gap','FontName','Times New Roman','FontSize',12);
xlabel(ax2,'Zealot fraction z','FontWeight','bold');
ylabel(ax2,'Selection intensity \beta','FontWeight','bold','Interpreter','tex');
title(ax2,'Monte Carlo branch gap: Stag Hunt','FontWeight','bold');
safe_save_png_2016a(fig2,'MC_SH_PRE_fine_branch_gap_2016a.png');

%% ============================================================
% CSV EXPORT
%% ============================================================
csvFile = 'MC_SH_PRE_fine_summary_2016a.csv';
fid = fopen(csvFile,'w');
if fid < 0
    warning('Could not open %s for writing.',csvFile);
else
    fprintf(fid,'beta,z,xLowMean,xLowCI95,xHighMean,xHighCI95,gapMean,gapCI95,fracLowAbsorbed\n');
    for b = 1:nB
        for iz = 1:nZ
            fprintf(fid,'%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n', ...
                betaList(b),zList(iz),xLowMean(b,iz),xLowCI95(b,iz), ...
                xHighMean(b,iz),xHighCI95(b,iz),gapMean(b,iz),gapCI95(b,iz), ...
                fracAbsLow(b,iz));
        end
    end
    fclose(fid);
end

%% ============================================================
% FINAL MAT FILE
%% ============================================================
save('MC_SH_PRE_fine_results_2016a.mat','p','betaList','zList','nRep', ...
    'maxSweeps','burnSweeps','sampleEvery','swapFactor','xLowAll','xHighAll', ...
    'absorbedLowAll','absorbedHighAll','elapsedByRep','totalRuntime', ...
    'xLowMean','xHighMean','xLowCI95','xHighCI95','gapAll','gapMean','gapCI95', ...
    'fracAbsLow','gapTol','mergeTol','highCoopTol','minBiPoints', ...
    'hasPositiveBistability','hasTrueCME','hasRapidSmoothFull','zCritCME','zFirstHigh');

fprintf('Saved: MC_SH_PRE_fine_branch_curves_2016a.png, MC_SH_PRE_fine_branch_gap_2016a.png,\n');
fprintf('       %s, MC_SH_PRE_fine_results_2016a.mat, %s\n',csvFile,checkpointFile);
fprintf('Finished.\n');

end

%% ============================================================
% SWITCHING PROBABILITIES (same kernels as the PA model)
%% ============================================================
function [probD2C,probC2D] = precompute_switch_probs_2016a(beta,p)
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
    Delta = A*(p.R-p.T) + B*(p.S-p.P);
    probD2C(nCoop+1) = p.mu + (1-p.mu)*A*fermi_2016a( beta*Delta);
    probC2D(nCoop+1) = p.mu + (1-p.mu)*B*fermi_2016a(-beta*Delta);
end

end

%% ============================================================
% ONE MONTE CARLO RUN
%% ============================================================
function [xMean,hitFull] = run_MC_branch_2016a(state,neighbors,probD2C,probC2D,p, ...
    maxSweeps,burnSweeps,sampleEvery,z)
% Cooperative-neighbour counts are maintained incrementally (zealots count
% as cooperators); they are updated only when a node changes strategy.
% Outputs: xMean is the time average of x over the measurement window
% (1 if full ordinary cooperation is reached, 0 if full defection is
% reached when z = 0, NaN if there are no ordinary players), and hitFull
% is true if full ordinary cooperation was reached.

N = p.N;
k = p.k;

ordinaryCount = sum(state ~= 2);
ordinaryCoop  = sum(state == 1);
hitFull       = false;

if ordinaryCount == 0
    xMean = NaN;
    return;
end

isCoop  = (state > 0);
coopNbr = zeros(N,1);
for jj = 1:k
    coopNbr = coopNbr + isCoop(neighbors(:,jj));
end

maxSamples    = floor((maxSweeps-burnSweeps)/sampleEvery) + 2;
sampleVals    = zeros(maxSamples,1);
sampleCounter = 0;

for sweep = 1:maxSweeps
    for attempt = 1:N
        i = randi(N);
        u = rand;
        if state(i) == 2                       % zealots never update
            continue;
        end
        nC = coopNbr(i);
        if state(i) == 0
            if u < probD2C(nC+1)               % D -> C
                state(i) = 1;
                ordinaryCoop = ordinaryCoop + 1;
                nb = neighbors(i,:);
                coopNbr(nb) = coopNbr(nb) + 1;
            end
        else
            if u < probC2D(nC+1)               % C -> D
                state(i) = 0;
                ordinaryCoop = ordinaryCoop - 1;
                nb = neighbors(i,:);
                coopNbr(nb) = coopNbr(nb) - 1;
            end
        end
    end

    if p.mu == 0
        % Full ordinary cooperation is absorbing when mu = 0.
        if ordinaryCoop == ordinaryCount
            xMean   = 1;
            hitFull = true;
            return;
        end
        % Without zealots, full defection is absorbing as well.
        if z == 0 && ordinaryCoop == 0
            xMean = 0;
            return;
        end
    end

    if sweep > burnSweeps && mod(sweep,sampleEvery) == 0
        sampleCounter = sampleCounter + 1;
        sampleVals(sampleCounter) = ordinaryCoop/ordinaryCount;
    end
end

if sampleCounter > 0
    xMean = mean(sampleVals(1:sampleCounter));
else
    xMean = ordinaryCoop/ordinaryCount;
end

end

%% ============================================================
% INITIAL BRANCH STATES
%% ============================================================
function state = initialize_branch_state_2016a(N,z,zealotOrder,branchType,p)
% Zealots: the first round(z*N) nodes of zealotOrder (nested sets in z).
% branchType 'LOW' starts with a fraction p.xLow0 of ordinary
% cooperators and 'HIGH' with p.xHigh0; at least one ordinary
% cooperator and one ordinary defector are kept.

state = zeros(N,1);
nZ = round(z*N);
state(zealotOrder(1:nZ)) = 2;

ordinaryNodes = find(state ~= 2);
nOrd = length(ordinaryNodes);
if nOrd <= 1
    return;
end

if strcmp(branchType,'LOW')
    nC0 = round(p.xLow0*nOrd);
    nC0 = min(max(1,nC0),nOrd-1);            % at least one ordinary cooperator
elseif strcmp(branchType,'HIGH')
    nD0 = round((1-p.xHigh0)*nOrd);
    nD0 = min(max(1,nD0),nOrd-1);            % at least one ordinary defector
    nC0 = nOrd - nD0;
else
    error('Unknown branch type.');
end

permOrd = ordinaryNodes(randperm(nOrd));
state(permOrd(1:nC0)) = 1;

end

%% ============================================================
% RANDOM k-REGULAR GRAPH BY DEGREE-PRESERVING DOUBLE-EDGE SWAPS
%% ============================================================
function neighbors = random_regular_neighbors_switching_2016a(N,k,swapFactor)
% Ring lattice (k/2 neighbours on each side) followed by swapFactor*M
% successful swaps (a,b),(c,d) -> (a,d),(c,b), with a random orientation of
% the second edge; self-loops and multiple edges are rejected.
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
    A = false(N,N);
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
    if rand < 0.5
        c = edges(e2,1);  d = edges(e2,2);
    else
        c = edges(e2,2);  d = edges(e2,1);
    end
    if a == d || c == b
        continue;                             % self-loop
    end
    if A(a,d) || A(c,b)
        continue;                             % multiple edge
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
% LOGISTIC (FERMI) FUNCTION
%% ============================================================
function y = fermi_2016a(a)
% Numerically stable logistic function 1/(1+exp(-a)).

if a >= 0
    y = 1/(1+exp(-a));
else
    ea = exp(a);
    y = ea/(1+ea);
end

end

%% ============================================================
% 95% CONFIDENCE BAND
%% ============================================================
function plot_CI_band_2016a(ax,x,meanY,ciY,bandColor)
% Shaded band meanY +/- ciY on the axes ax, limited to [0, 1.02];
% NaN points are skipped.

x     = x(:).';
meanY = meanY(:).';
ciY   = ciY(:).';
ok    = ~isnan(meanY) & ~isnan(ciY);
x = x(ok); meanY = meanY(ok); ciY = ciY(ok);
if numel(x) < 2
    return;
end
upperY = min(meanY + ciY,1.02);
lowerY = max(meanY - ciY,0);
fill([x fliplr(x)],[upperY fliplr(lowerY)],bandColor,'EdgeColor','none','Parent',ax);

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
