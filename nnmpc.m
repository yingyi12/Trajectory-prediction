function [t, x, u,xest,yest,xxpossible1,yypossible1,xxpossible2,yypossible2,xxpossible3,yypossible3,xp,ep,m,n,EVxpre,EVypre] = nnmpc(runningcosts, terminalcosts, ...
              constraints, terminalconstraints, ...
              linearconstraints, system, ...
              mpciterations, N, T, tmeasure, xmeasure, u0, ...
              varargin)
% nmpc(runningcosts, terminalcosts, constraints, ...
%      terminalconstraints, linearconstraints, system, ...
%      mpciterations, N, T, tmeasure, xmeasure, u0, ...
%      tol_opt, opt_option, ...
%      type, atol_ode_real, rtol_ode_real, atol_ode_sim, rtol_ode_sim, ...
%      iprint, printHeader, printClosedloopData, plotTrajectories)
% Computes the closed loop solution for the NMPC problem defined by
% the functions
%   runningcosts:         evaluates the running costs for state and control
%                         at one sampling instant.
%                         The function returns the running costs for one
%                         sampling instant.
%          Usage: [cost] = runningcosts(t, x, u)
%                 with time t, state x and control u
%   terminalcosts:        evaluates the terminal costs for state at the end
%                         of the open loop horizon.
%                         The function returns value of the terminal costs.
%          Usage: cost = terminalcosts(t, x)
%                 with time t and state x
%   constraints:          computes the value of the restrictions for a
%                         sampling instance provided the data t, x and u
%                         given by the optimization method.
%                         The function returns the value of the
%                         restrictions for a sampling instance separated
%                         for inequality restrictions c and equality
%                         restrictions ceq.
%          Usage: [c,ceq] = constraints(t, x, u)
%                 with time t, state x and control u
%   terminalconstraints:  computes the value of the terminal restrictions
%                         provided the data t, x and u given by the
%                         optimization method.
%                         The function returns the value of the
%                         terminal restriction for inequality restrictions
%                         c and equality restrictions ceq.
%          Usage: [c,ceq] = terminalconstraints(t, x)
%                 with time t and state x
%   linearconstraints:    sets the linear constraints of the discretized
%                         optimal control problem. This is particularly
%                         useful to set control and state bounds.
%                         The function returns the required matrices for
%                         the linear inequality and equality constraints A
%                         and Aeq, the corresponding right hand sides b and
%                         beq as well as the lower and upper bound of the
%                         control.
%          Usage: [A, b, Aeq, beq, lb, ub] = linearconstraints(t, x, u)
%                 with time t, state x and control u
%   system:               evaluates the difference equation describing the
%                         process given time t, state vector x and control
%                         u.
%                         The function returns the state vector x at the
%                         next time instant.
%          Usage: [y] = system(t, x, u, T)
%                 with time t, state x, control u and sampling interval T
% for a given number of NMPC iteration steps (mpciterations). For
% the open loop problem, the horizon is defined by the number of
% time instances N and the sampling time T. Note that the dynamic
% can also be the solution of a differential equation. Moreover, the
% initial time tmeasure, the state measurement xmeasure and a guess of
% the optimal control u0 are required.
%
% Arguments:
%   mpciterations:  Number of MPC iterations to be performed
%   N:              Length of optimization horizon
%   T:              Sampling interval
%   tmeasure:       Time measurement of initial value
%   xmeasure:       State measurement of initial value
%   u0:             Initial guess of open loop control
%
% Optional arguments:
%       iprint:     = 0  Print closed loop data(default)
%                   = 1  Print closed loop data and errors of the
%                        optimization method
%                   = 2  Print closed loop data and errors and warnings of
%                        the optimization method
%                   >= 5 Print closed loop data and errors and warnings of
%                        the optimization method as well as graphical
%                        output of closed loop state trajectories
%                   >=10 Print closed loop data and errors and warnings of
%                        the optimization method with error and warning
%                        description
%   printHeader:         Clarifying header for selective output of closed
%                        loop data, cf. printClosedloopData
%   printClosedloopData: Selective output of closed loop data
%   plotTrajectories:    Graphical output of the trajectories, requires
%                        iprint >= 4
%       tol_opt:         Tolerance of the optimization method
%       opt_option: = 0: Active-set method used for optimization (default)
%                   = 1: Interior-point method used for optimization
%                   = 2: Trust-region reflective method used for
%                        optimization
%   type:                Type of dynamic, either difference equation or
%                        differential equation can be used
%    atol_ode_real:      Absolute tolerance of the ODE solver for the
%                        simulated process
%    rtol_ode_real:      Relative tolerance of the ODE solver for the
%                        simulated process
%    atol_ode_sim:       Absolute tolerance of the ODE solver for the
%                        simulated NMPC prediction
%    rtol_ode_sim:       Relative tolerance of the ODE solver for the
%                        simulated NMPC prediction
%
% Internal Functions:
%   measureInitialValue:          measures the new initial values for t0
%                                 and x0 by adopting values computed by
%                                 method applyControl.
%                                 The function returns new initial state
%                                 vector x0 at sampling instant t0.
%   applyControl:                 applies the first control element of u to
%                                 the simulated process for one sampling
%                                 interval T.
%                                 The function returns closed loop state
%                                 vector xapplied at sampling instant
%                                 tapplied.
%   shiftHorizon:                 applies the shift method to the open loop
%                                 control in order to ease the restart.
%                                 The function returns a new initial guess
%                                 u0 of the control.
%   solveOptimalControlProblem:   solves the optimal control problem of the
%                                 horizon N with sampling length T for the
%                                 given initial values t0 and x0 and the
%                                 initial guess u0 using the specified
%                                 algorithm.
%                                 The function returns the computed optimal
%                                 control u, the corresponding value of the
%                                 cost function V as well as possible exit
%                                 flags and additional output of the
%                                 optimization method.
%   costfunction:                 evaluates the cost function of the
%                                 optimal control problem over the horizon
%                                 N with sampling time T for the current
%                                 data of the optimization method t0, x0
%                                 and u.
%                                 The function return the computed cost
%                                 function value.
%   nonlinearconstraints:         computes the value of the restrictions
%                                 for all sampling instances provided the
%                                 data t0, x0 and u given by the
%                                 optimization method.
%                                 The function returns the value of the
%                                 restrictions for all sampling instances
%                                 separated for inequality restrictions c
%                                 and equality restrictions ceq.
%   computeOpenloopSolution:      computes the open loop solution over the
%                                 horizon N with sampling time T for the
%                                 initial values t0 and x0 as well as the
%                                 control u.
%                                 The function returns the complete open
%                                 loop solution over the requested horizon.
%   dynamic:                      evaluates the dynamic of the system for
%                                 given initial values t0 and x0 over the
%                                 interval [t0, tf] using the control u.
%                                 The function returns the state vector x
%                                 at time instant tf as well as an output
%                                 of all intermediate evaluated time
%                                 instances.
%   printSolution:                prints out information on the current MPC
%                                 step, in particular state and control
%                                 information as well as required computing
%                                 times and exitflags/outputs of the used
%                                 optimization method. The flow of
%                                 information can be controlled by the
%                                 variable iprint and the functions
%                                 printHeader, printClosedloopData and
%                                 plotTrajectories.
%
% Version of May 30, 2011, in which a bug appearing in the case of 
% multiple constraints has been fixed
%
% (C) Lars Gruene, Juergen Pannek 2011

    if (nargin>=13)
        tol_opt = varargin{1};
    else
        tol_opt = 1e-6;
        end;
    if (nargin>=14)
        opt_option = varargin{2};
    else
        opt_option = 0;
    end;
    if (nargin>=15)
        if ( strcmp(varargin{3}, 'difference equation') || ...
                strcmp(varargin{3}, 'differential equation') )
            type = varargin{3};
        else
            fprintf([' Wrong input for type of dynamic: use either ', ...
                '"difference equation" or "differential equation".']);
        end
    else
        type = 'difference equation';
    end;
    if (nargin>=16)
        atol_ode_real = varargin{4};
    else
        atol_ode_real = 1e-8;
    end;
    if (nargin>=17)
        rtol_ode_real = varargin{5};
    else
        rtol_ode_real = 1e-8;
    end;
    if (nargin>=18)
        atol_ode_sim = varargin{6};
    else
        atol_ode_sim = atol_ode_real;
    end;
    if (nargin>=19)
        rtol_ode_sim = varargin{7};
    else
        rtol_ode_sim = rtol_ode_real;
    end;
    if (nargin>=20)
        iprint = varargin{8};
    else
        iprint = 0;
    end;
    if (nargin>=21)
        printHeader = varargin{9};
    else
        printHeader = @printHeaderDummy;
    end;
    if (nargin>=22)
        printClosedloopData = varargin{10};
    else
        printClosedloopData = @printClosedloopDataDummy;
    end;
    if (nargin>=23)
        plotTrajectories = varargin{11};
    else
        plotTrajectories = @plotTrajectoriesDummy;
    end;

    % Determine MATLAB Version and
    % specify and configure optimization method
    vs = version('-release');
    vyear = str2num(vs(1:4));
    if (vyear <= 2007)
        fprintf('MATLAB version R2007 or earlier detected\n');
        if ( opt_option == 0 )
            options = optimset('Display','off',...
                'TolFun', tol_opt,...
                'MaxIter', 2000,...
                'LargeScale', 'off',...
                'RelLineSrchBnd', [],...
                'RelLineSrchBndDuration', 1);
        elseif ( opt_option == 1 )
            error('nmpc:WrongArgument', '%s\n%s', ...
                  'Interior point method not supported in MATLAB R2007', ...
                  'Please use opt_option = 0 or opt_option = 2');
        elseif ( opt_option == 2 )
             options = optimset('Display','off',...
                 'TolFun', tol_opt,...
                 'MaxIter', 2000,...
                 'LargeScale', 'on',...
                 'Hessian', 'off',...
                 'MaxPCGIter', max(1,floor(size(u0,1)*size(u0,2)/2)),...
                 'PrecondBandWidth', 0,...
                 'TolPCG', 1e-1);
        end
    else
        fprintf('MATLAB version R2008 or newer detected\n');
        if ( opt_option == 0 )
            options = optimset('Display','off',...
                'TolFun', tol_opt,...
                'MaxIter', 10000,...
                'Algorithm', 'active-set',...
                'FinDiffType', 'forward',...
                'RelLineSrchBnd', [],...
                'RelLineSrchBndDuration', 1,...
                'TolConSQP', 1e-6);
        elseif ( opt_option == 1 )
            options = optimset('Display','off',...
                'TolFun', tol_opt,...
                'MaxIter', 2000,...
                'Algorithm', 'interior-point',...
                'AlwaysHonorConstraints', 'bounds',...
                'FinDiffType', 'forward',...
                'HessFcn', [],...
                'Hessian', 'bfgs',...
                'HessMult', [],...
                'InitBarrierParam', 0.1,...
                'InitTrustRegionRadius', sqrt(size(u0,1)*size(u0,2)),...
                'MaxProjCGIter', 2*size(u0,1)*size(u0,2),...
                'ObjectiveLimit', -1e20,...
                'ScaleProblem', 'obj-and-constr',...
                'SubproblemAlgorithm', 'cg',...
                'TolProjCG', 1e-2,...
                'TolProjCGAbs', 1e-10);
        %                       'UseParallel','always',...
        elseif ( opt_option == 2 )
            options = optimset('Display','off',...
                'TolFun', tol_opt,...
                'MaxIter', 2000,...
                'Algorithm', 'trust-region-reflective',...
                'Hessian', 'off',...
                'MaxPCGIter', max(1,floor(size(u0,1)*size(u0,2)/2)),...
                'PrecondBandWidth', 0,...
                'TolPCG', 1e-1);
        end
    end
%  
 options.Algorithm = 'sqp'; 
%soft constraint, allow state constraint violation 
    warning off all
    t = [];
    x = [];
    u = [];
    

T = 1;




[xx1,yy1] = part1;
[xx2,yy2] = part2;
[xx3,yy3] = part3;
[xx4,yy4] = part4;
A=[0.90 0.08 0.02;
   0.08 0.90 0.02;
   0.08 0.02 0.90];    
F{1}= [1 1 0 0 0 0;
       0 1 0 0 0 0;
       0 0 0 0 0 0;
       0 0 0 1 0 0;
       0 0 0 0 1 0;
       0 0 0 0 0 0];
   
   
   
F{2}= [1 0.1 0 0 0 0;
       0 1 0 0 0 0;
       0 0 0 0 0 0;
       0 0 0 1 1 0;
       0 0 0 0 1 0;
       0 0 0 0 0 0]; 

F{3}= [1 0.1 0 0 0 0;
       0 1 0 0 0 0;
       0 0 0 0 0 0;
       0 0 0 1 -1 0;
       0 0 0 0 1 0;
       0 0 0 0 0 0];

% white noise matrix
% Q{1} =0.1*[0 0 0 0 0 0;
%            0 0 0 0 0 0;
%            0 0 1 0 0 0;
%            0 0 0 0 0 0;
%            0 0 0 0 0 0;
%            0 0 0 0 0 1];
Q{1} =0.01*eye(6);
% 
% Q{2}=0.1*[0 0 0 0 0 0;
%            0 0 0 0 0 0;
%            0 0 1 0 0 0;
%            0 0 0 0 0 0;
%            0 0 0 0 0 0;
%            0 0 0 0 0 1];

Q{2} =0.01*eye(6);

% Q{3}=0.1*[0 0 0 0 0 0;
%            0 0 0 0 0 0;
%            0 0 1 0 0 0;
%            0 0 0 0 0 0;
%            0 0 0 0 0 0;
%            0 0 0 0 0 1];
Q{3} =0.01*eye(6);

H = [1 0 0 0 0 0 ;
     0 0 0 1 0 0];
R = eye(2);
u1 = [0,0.5,0.5];
u2 = [0,0.5,0.5];
u3 = [0.4,0.3,0.3];
u4 = [0.4,0.3,0.3];
xn1 = cell(3,1);
pn1 = cell(3,1);
xn2 = cell(3,1);
pn2 = cell(3,1);
xn3 = cell(3,1);
pn3 = cell(3,1);
xn4 = cell(3,1);
pn4 = cell(3,1);

B = 0.8*eye(6);
c1=3;
c2=3;
N  = 10;
% Start of the NMPC iteration
    mpciter = 0;
    while(mpciter < mpciterations)
        l = mpciter+1;
        time(l)=l*1;
        if l == 1
            for i =1:3
                xn1{i}=[165;1;0;75;1;0];
                pn1{i}=0.4*eye(6);
                xn2{i}=[40;1;0;-17;1;0];
                pn2{i}=0.4*eye(6);     
                xn3{i}=[40;1;0;7;1;0];
                pn3{i}=0.4*eye(6);
                xn4{i}=[30;1;0;6;1;0];
                pn4{i}=0.4*eye(6);
            end
        end

        

        y1 = [xx1(l);yy1(l)];

            [x1,p1,xn1,pn1,u1]=imm(F,H,Q,R,A,xn1,pn1,u1,y1);

        for t = 1:N
            possible1{l,1} = x1;
            possible1{l,t+1} = F{1}*possible1{l,t};
            L = possible1{l,t};
            xpossible1(l,t) = L(1);
            ypossible1(l,t) = L(4);

            xtv1{l,1} = p1;
            xtv1{l,t+1} = F{1}*xtv1{l,t}*F{1}'+B*Q{1}*B';
            Var = diag(xtv1{l,t});
            kappa11 = -1*log(1-u1(1));
            mpossible11(l,t) = sqrt(Var(1)*kappa11);
            npossible11(l,t) = sqrt(Var(4)*kappa11);
        end

        for t = 1:N
            possible2{l,1} = x1;
            possible2{l,t+1} = F{2}*possible2{l,t};
            L = possible2{l,t};
            xpossible2(l,t) = L(1);
            ypossible2(l,t) = L(4);

            xtv2{l,1} = p1;
            xtv2{l,t+1} = F{2}*xtv2{l,t}*F{2}'+B*Q{2}*B';
            Var = diag(xtv2{l,t});
            kappa21 = -1*log(1-u1(2));
            mpossible21(l,t) = sqrt(Var(1)*kappa21);
            npossible21(l,t) = sqrt(Var(4)*kappa21);
        end
        for t = 1:N
            possible3{l,1} = x1;
            possible3{l,t+1} = F{3}*possible3{l,t};
            L = possible3{l,t};
            xpossible3(l,t) = L(1);
            ypossible3(l,t) = L(4);

            xtv3{l,1} = p1;
            xtv3{l,t+1} = F{3}*xtv3{l,t}*F{3}'+B*Q{3}*B';
            Var = diag(xtv3{l,t});
            kappa31 = -1*log(1-u1(3));
            mpossible31(l,t) = sqrt(Var(1)*kappa31);
            npossible31(l,t) = sqrt(Var(4)*kappa31);
        end
        xest(1,l)=x1(1);
        yest(1,l)=x1(4);
        xxpossible1{1,l}=xpossible1(l,:) ;%j participant k step
        yypossible1{1,l}=ypossible1(l,:) ;
        xxpossible2{1,l}=xpossible2(l,:) ;
        yypossible2{1,l}=ypossible2(l,:) ;
        xxpossible3{1,l}=xpossible3(l,:) ;
        yypossible3{1,l}=ypossible3(l,:) ;
        xp11 =xpossible1(l,:);% model 1
        yp11 =ypossible1(l,:);
        xp21 =xpossible2(l,:);%model 2
        yp21 =ypossible2(l,:);
        xp31 =xpossible3(l,:);% model 3
        yp31 =ypossible3(l,:);
        m11 = c1*mpossible11(l,:);
        m21 = c1*mpossible21(l,:);
        m31 = c1*mpossible31(l,:);
        n11 = c2*npossible11(l,:);
        n21 = c2*npossible21(l,:);
        n31 = c2*npossible31(l,:);
        

                y2 = [xx2(l);yy2(l)];
     
            [x2,p2,xn2,pn2,u2]=imm(F,H,Q,R,A,xn2,pn2,u2,y2);
       
        for t = 1:N
            possible1{l,1} = x2;
            possible1{l,t+1} = F{1}*possible1{l,t};
            L = possible1{l,t};
            xpossible1(l,t) = L(1);
            ypossible1(l,t) = L(4);

            xtv1{l,1} = p2;
            xtv1{l,t+1} = F{1}*xtv1{l,t}*F{1}'+B*Q{1}*B';
            Var = diag(xtv1{l,t});
            kappa12 = -1*log(1-u2(1));
            mpossible12(l,t) = sqrt(Var(1)*kappa12);
            npossible12(l,t) = sqrt(Var(4)*kappa12);
        end
        for t = 1:N
            possible2{l,1} = x2;
            possible2{l,t+1} = F{2}*possible2{l,t};
            L = possible2{l,t};
            xpossible2(l,t) = L(1);
            ypossible2(l,t) = L(4);

            xtv2{l,1} = p2;
            xtv2{l,t+1} = F{2}*xtv2{l,t}*F{2}'+B*Q{2}*B';
            Var = diag(xtv2{l,t});
            kappa22 = -1*log(1-u2(2));
            mpossible22(l,t) = sqrt(Var(1)*kappa22);
            npossible22(l,t) = sqrt(Var(4)*kappa22);
        end
        for t = 1:N
            possible3{l,1} = x2;
            possible3{l,t+1} = F{3}*possible3{l,t};
            L = possible3{l,t};
            xpossible3(l,t) = L(1);
            ypossible3(l,t) = L(4);

            xtv3{l,1} = p2;
            xtv3{l,t+1} = F{3}*xtv3{l,t}*F{3}'+B*Q{3}*B';
            Var = diag(xtv3{l,t});
            kappa32 = -1*log(1-u2(3));
            mpossible32(l,t) = sqrt(Var(1)*kappa32);
            npossible32(l,t) = sqrt(Var(4)*kappa32);
        end
        xest(2,l)=x2(1);
        yest(2,l)=x2(4);
        xxpossible1{2,l}=xpossible1(l,:) ;%j participant k step
        yypossible1{2,l}=ypossible1(l,:) ;
        xxpossible2{2,l}=xpossible2(l,:) ;
        yypossible2{2,l}=ypossible2(l,:) ;
        xxpossible3{2,l}=xpossible3(l,:) ;
        yypossible3{2,l}=ypossible3(l,:) ;
        xp12 =xpossible1(l,:);% model 1
        yp12 =ypossible1(l,:);
        xp22 =xpossible2(l,:);%model 2
        yp22 =ypossible2(l,:);
        xp32 =xpossible3(l,:);% model 3
        yp32 =ypossible3(l,:);
        m12 = c1*mpossible12(l,:);
        m22 = c1*mpossible22(l,:);
        m32 = c1*mpossible32(l,:);
        n12 = c2*npossible12(l,:);
        n22 = c2*npossible22(l,:);
        n32 = c2*npossible32(l,:); 

        

        y3= [xx3(l);yy3(l)];
     
            [x3,p3,xn3,pn3,u3]=imm(F,H,Q,R,A,xn3,pn3,u3,y3);
       
        for t = 1:N
            possible1{l,1} = x3;
            possible1{l,t+1} = F{1}*possible1{l,t};
            L = possible1{l,t};
            xpossible1(l,t) = L(1);
            ypossible1(l,t) = L(4);

            xtv1{l,1} = p3;
            xtv1{l,t+1} = F{1}*xtv1{l,t}*F{1}'+B*Q{1}*B';
            Var = diag(xtv1{l,t});
            kappa13 = -1*log(1-u3(1));
            mpossible13(l,t) = sqrt(Var(1)*kappa13);
            npossible13(l,t) = sqrt(Var(4)*kappa13);
        end
        for t = 1:N
            possible2{l,1} = x3;
            possible2{l,t+1} = F{2}*possible2{l,t};
            L = possible2{l,t};
            xpossible2(l,t) = L(1);
            ypossible2(l,t) = L(4);

            xtv2{l,1} = p3;
            xtv2{l,t+1} = F{2}*xtv2{l,t}*F{2}'+B*Q{2}*B';
            Var = diag(xtv2{l,t});
            kappa23 = -1*log(1-u3(2));
            mpossible23(l,t) = sqrt(Var(1)*kappa23);
            npossible23(l,t) = sqrt(Var(4)*kappa23);
        end
        for t = 1:N
            possible3{l,1} = x3;
            possible3{l,t+1} = F{3}*possible3{l,t};
            L = possible3{l,t};
            xpossible3(l,t) = L(1);
            ypossible3(l,t) = L(4);

             xtv3{l,1} = p3;
            xtv3{l,t+1} = F{3}*xtv3{l,t}*F{3}'+B*Q{3}*B';
            Var = diag(xtv3{l,t});
            kappa33 = -1*log(1-u3(3));
            mpossible33(l,t) = sqrt(Var(1)*kappa33);
            npossible33(l,t) = sqrt(Var(4)*kappa33);
        end
        xest(3,l)=x3(1);
        yest(3,l)=x3(4);
        xxpossible1{3,l}=xpossible1(l,:);%j participant k step
        yypossible1{3,l}=ypossible1(l,:);
        xxpossible2{3,l}=xpossible2(l,:);
        yypossible2{3,l}=ypossible2(l,:);
        xxpossible3{3,l}=xpossible3(l,:);
        yypossible3{3,l}=ypossible3(l,:);
        xp13 =xpossible1(l,:);% model 1
        yp13 =ypossible1(l,:);
        xp23 =xpossible2(l,:);%model 2
        yp23 =ypossible2(l,:);
        xp33 =xpossible3(l,:);% model 3
        yp33 =ypossible3(l,:);
        m13 = c1*mpossible13(l,:);
        m23 = c1*mpossible23(l,:);
        m33 = c1*mpossible33(l,:);
        n13 = c2*npossible13(l,:);
        n23 = c2*npossible23(l,:);
        n33 = c2*npossible33(l,:); 
             y4= [xx4(l);yy4(l)];
     
            [x4,p4,xn4,pn4,u4]=imm(F,H,Q,R,A,xn4,pn4,u4,y4);
       
        for t = 1:N
            possible1{l,1} = x4;
            possible1{l,t+1} = F{1}*possible1{l,t};
            L = possible1{l,t};
            xpossible1(l,t) = L(1);
            ypossible1(l,t) = L(4);

            xtv1{l,1} = p4;
            xtv1{l,t+1} = F{1}*xtv1{l,t}*F{1}'+B*Q{1}*B';
            Var = diag(xtv1{l,t});
            kappa14 = -1*log(1-u4(1));
            mpossible14(l,t) = sqrt(Var(1)*kappa14);
            npossible14(l,t) = sqrt(Var(4)*kappa14);
        end
        for t = 1:N
            possible2{l,1} = x4;
            possible2{l,t+1} = F{2}*possible2{l,t};
            L = possible2{l,t};
            xpossible2(l,t) = L(1);
            ypossible2(l,t) = L(4);

            xtv2{l,1} = p4;
            xtv2{l,t+1} = F{2}*xtv2{l,t}*F{2}'+B*Q{2}*B';
            Var = diag(xtv2{l,t});
            kappa24 = -1*log(1-u4(2));
            mpossible24(l,t) = sqrt(Var(1)*kappa24);
            npossible24(l,t) = sqrt(Var(4)*kappa24);
        end
        for t = 1:N
            possible3{l,1} = x4;
            possible3{l,t+1} = F{3}*possible3{l,t};
            L = possible3{l,t};
            xpossible3(l,t) = L(1);
            ypossible3(l,t) = L(4);

             xtv3{l,1} = p4;
            xtv3{l,t+1} = F{3}*xtv3{l,t}*F{3}'+B*Q{3}*B';
            Var = diag(xtv3{l,t});
            kappa34 = -1*log(1-u4(3));
            mpossible34(l,t) = sqrt(Var(1)*kappa34);
            npossible34(l,t) = sqrt(Var(4)*kappa34);
        end
        xest(4,l)=x4(1);
        yest(4,l)=x4(4);
        xxpossible1{4,l}=xpossible1(l,:);%j participant k step
        yypossible1{4,l}=ypossible1(l,:);
        xxpossible2{4,l}=xpossible2(l,:);
        yypossible2{4,l}=ypossible2(l,:);
        xxpossible3{4,l}=xpossible3(l,:);
        yypossible3{4,l}=ypossible3(l,:);
        xp14 =xpossible1(l,:);% model 1
        yp14 =ypossible1(l,:);
        xp24 =xpossible2(l,:);%model 2
        yp24 =ypossible2(l,:);
        xp34 =xpossible3(l,:);% model 3
        yp34 =ypossible3(l,:);
        m14 = c1*mpossible14(l,:);
        m24 = c1*mpossible24(l,:);
        m34 = c1*mpossible34(l,:);
        n14 = c2*npossible14(l,:);
        n24 = c2*npossible24(l,:);
        n34 = c2*npossible34(l,:); 


        xp = [xp11;xp12;xp13;xp14;xp21;xp22;xp23;xp24;xp31;xp32;xp33;xp34;yp11;yp12;yp13;yp14;yp21;yp22;yp23;yp24;yp31;yp32;yp33;yp34];
        xp = xp';
        m =  [m11;m12;m13;m14;m21;m22;m23;m24;m31;m32;m33;m34];
        n = [n11;n12;n13;n14;n21;n22;n23;n24;n31;n32;n33;n34];
        m = m';
        n = n';

        th = linspace(0,2*pi) ;

% 
        x1 = xp(:,1)+m(:,1).*cos(th) ;
        y1 = xp(:,13)+n(:,1).*sin(th) ;
        x2 = xp(:,2)+m(:,2).*cos(th) ;
        y2 = xp(:,14)+n(:,2).*sin(th) ;
        x3 = xp(:,3)+m(:,3).*cos(th) ;
        y3 = xp(:,15)+n(:,3).*sin(th) ;
        x4 = xp(:,4)+m(:,4).*cos(th) ;
        y4 = xp(:,16)+n(:,4).*sin(th) ;
        x5 = xp(:,5)+m(:,5).*cos(th) ;
        y5 = xp(:,17)+n(:,5).*sin(th) ;
        x6 = xp(:,6)+m(:,6).*cos(th) ;
        y6 = xp(:,18)+n(:,6).*sin(th) ;
        x7 = xp(:,7)+m(:,7).*cos(th) ;
        y7 = xp(:,19)+n(:,7).*sin(th) ;
        x8 = xp(:,8)+m(:,8).*cos(th) ;
        y8 = xp(:,20)+n(:,8).*sin(th) ;
        x9 = xp(:,9)+m(:,9).*cos(th) ;
        y9 = xp(:,21)+n(:,9).*sin(th) ;
        x10 = xp(:,10)+m(:,10).*cos(th) ;
        y10 = xp(:,22)+n(:,10).*sin(th) ;
        x11 = xp(:,11)+m(:,11).*cos(th) ;
        y11 = xp(:,23)+n(:,11).*sin(th) ;
        x12 = xp(:,12)+m(:,12).*cos(th) ;
        y12 = xp(:,24)+n(:,12).*sin(th) ;
        ep{1,l} = x1';
        ep{2,l} = y1';
        ep{3,l} = x2';
        ep{4,l} = y2';
        ep{5,l} = x3';
        ep{6,l} = y3';
        ep{7,l} = x4';
        ep{8,l} = y4';
        ep{9,l} = x5';
        ep{10,l} = y5';
        ep{11,l} = x6';
        ep{12,l} = y6';
        ep{13,l} = x7';
        ep{14,l} = y7';
        ep{15,l} = x8';
        ep{16,l} = y8';
        ep{17,l} = x9';
        ep{18,l} = y9';
        ep{19,l} = x10';
        ep{20,l} = y10';
        ep{21,l} = x11';
        ep{22,l} = y11';
        ep{23,l} = x12';
        ep{24,l} = y12';

     
%         Step (1) of the NMPC algorithm:
%           Obtain new initial value
        [t0, x0] = measureInitialValue ( tmeasure, xmeasure );
%         Step (2) of the NMPC algorithm:
%           Solve the optimal control problem
        t_Start = tic;
        [u_new, V_current, exitflag, output,xpre] = solveOptimalControlProblem ...
            (runningcosts, terminalcosts, constraints, ...
            terminalconstraints, linearconstraints, system, ...
            N, t0, x0, u0, T,  ...
            atol_ode_sim, rtol_ode_sim, tol_opt, options, type,xp,m,n);
       EVxpre(l,:)=xpre(1:10,1);
       EVypre(l,:)=xpre(1:10,3);
        t_Elapsed = toc( t_Start );
%           Print solution
        if ( iprint >= 1 )
            printSolution(system, printHeader, printClosedloopData, ...
                          plotTrajectories, mpciter, T, t0, x0, u_new, ...
                          atol_ode_sim, rtol_ode_sim, type, iprint, ...
                          exitflag, output, t_Elapsed);
        end
%           Store closed loop data
        t = [ t; tmeasure ];
        x = [ x; xmeasure ];
        u = [ u; u_new(:,1) ];
%           Prepare restart
        u0 = shiftHorizon(u_new);
%         Step (3) of the NMPC algorithm:
%           Apply control to process
        [tmeasure, xmeasure] = applyControl(system, T, t0, x0, u_new, ...
            atol_ode_real, rtol_ode_real, type);
        mpciter = mpciter+1;
    end

    
end




function [t0, x0] = measureInitialValue ( tmeasure, xmeasure )
    t0 = tmeasure;
    x0 = xmeasure;
end

function [tapplied, xapplied] = applyControl(system, T, t0, x0, u, ...
                                atol_ode_real, rtol_ode_real, type)
    xapplied = dynamic(system, T, t0, x0, u(:,1), ...
                       atol_ode_real, rtol_ode_real, type);
    tapplied = t0+T;
end

function u0 = shiftHorizon(u)
    u0 = [u(:,2:size(u,2)) u(:,size(u,2))];
end

function [u, V, exitflag, output,x] = solveOptimalControlProblem ...
    (runningcosts, terminalcosts, constraints, terminalconstraints, ...
    linearconstraints, system, N, t0, x0, u0, T, ...
    atol_ode_sim, rtol_ode_sim, tol_opt, options, type,xp,m,n)
    x = zeros(N+1, length(x0));
    x = computeOpenloopSolution(system, N, T, t0, x0, u0, ...
                                atol_ode_sim, rtol_ode_sim, type);

    % Set control and linear bounds
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    for k=1:N
        [Anew, bnew, Aeqnew, beqnew, lbnew, ubnew] = ...
               linearconstraints(t0+k*T,x(k,:),u0(:,k));
        A = blkdiag(A,Anew);
        b = [b, bnew];
        Aeq = blkdiag(Aeq,Aeqnew);
        beq = [beq, beqnew];
        lb = [lb, lbnew];
        ub = [ub, ubnew];
    end
%     xp 
%     keyboard()
    % Solve optimization problem
    [u, V, exitflag, output] = fmincon(@(u) costfunction(runningcosts, ...
        terminalcosts, system, N, T, t0, x0, ...
        u, atol_ode_sim, rtol_ode_sim, type), u0, A, b, Aeq, beq, lb, ...
        ub, @(u) nonlinearconstraints(constraints, terminalconstraints, ...
        system, N, T, t0, x0, u, ...
        atol_ode_sim, rtol_ode_sim, type, xp,m,n), options);
%   if exitflag <0
%         keyboard()
%     end  
end

function cost = costfunction(runningcosts, terminalcosts, system, ...
                    N, T, t0, x0, u, ...
                    atol_ode_sim, rtol_ode_sim, type)
    cost = 0;
    x = zeros(N+1, length(x0));
    x = computeOpenloopSolution(system, N, T, t0, x0, u, ...
                                atol_ode_sim, rtol_ode_sim, type);

    for k=1:N
        cost = cost+runningcosts(t0+k*T, x(k,:), u(:,k));
    end
    cost = cost+terminalcosts(t0+(N+1)*T, x(N+1,:));
end
   
function [c,ceq] = nonlinearconstraints(constraints, ...
    terminalconstraints, system, ...
    N, T, t0, x0, u, atol_ode_sim, rtol_ode_sim, type,xp,m,n)
% 
    x = zeros(N+1, length(x0));
    x = computeOpenloopSolution(system, N, T, t0, x0, u, ...
                                atol_ode_sim, rtol_ode_sim, type);
    c = [];
    ceq = [];
    for k=1:N
        [cnew, ceqnew] = constraints(t0+k*T,x(k,:),u(:,k),xp(k,:),m(k,:),n(k,:));
        c = [c cnew];
        ceq = [ceq ceqnew];
    end
    [cnew, ceqnew] = terminalconstraints(t0+(N+1)*T,x(N+1,:),xp(N,:),m(N,:),n(N,:));
    c = [c cnew];
    ceq = [ceq ceqnew];
%     if x0(1)>20
%         keyboard()
%     end

end

function x = computeOpenloopSolution(system, N, T, t0, x0, u, ...
                                     atol_ode_sim, rtol_ode_sim, type)
    x(1,:) = x0;

    for k=1:N
        x(k+1,:) = dynamic(system, T, t0, x(k,:), u(:,k), ...
                             atol_ode_sim, rtol_ode_sim, type);
    end
end

function [x, t_intermediate, x_intermediate] = dynamic(system, T, t0, ...
             x0, u, atol_ode, rtol_ode, type)
    if ( strcmp(type, 'difference equation') )
        x = system(t0, x0, u, T);
        x_intermediate = [x0; x];
        t_intermediate = [t0, t0+T];
    elseif ( strcmp(type, 'differential equation') )
        options = odeset('AbsTol', atol_ode, 'RelTol', rtol_ode);
        [t_intermediate,x_intermediate] = ode45(system, ...
            [t0, t0+T], x0, options, u);
        x = x_intermediate(size(x_intermediate,1),:);
    end
end


% 
function printSolution(system, printHeader, printClosedloopData, ...
             plotTrajectories, mpciter, T, t0, x0, u, ...
             atol_ode, rtol_ode, type, iprint, exitflag, output, t_Elapsed)
    if (mpciter == 0)
        printHeader();
    end
    printClosedloopData(mpciter, u, x0, t_Elapsed);
    switch exitflag
        case -2
        if ( iprint >= 1 && iprint < 10 )
            fprintf(' Error F\n');
        elseif ( iprint >= 10 )
            fprintf(' Error: No feasible point was found\n')
        end
        case -1
        if ( iprint >= 1 && iprint < 10 )
            fprintf(' Error OT\n');
        elseif ( iprint >= 10 )
            fprintf([' Error: The output function terminated the',...
                     ' algorithm\n'])
        end
        case 0
        if ( iprint == 1 )
            fprintf('\n');
        elseif ( iprint >= 2 && iprint < 10 )
            fprintf(' Warning IT\n');
        elseif ( iprint >= 10 )
            fprintf([' Warning: Number of iterations exceeded',...
                     ' options.MaxIter or number of function',...
                     ' evaluations exceeded options.FunEvals\n'])
        end
        case 1
        if ( iprint == 1 )
            fprintf('\n');
        elseif ( iprint >= 2 && iprint < 10 )
            fprintf(' \n');
        elseif ( iprint >= 10 )
            fprintf([' First-order optimality measure was less',...
                     ' than options.TolFun, and maximum constraint',...
                     ' violation was less than options.TolCon\n'])
        end
        case 2
        if ( iprint == 1 )
            fprintf('\n');
        elseif ( iprint >= 2 && iprint < 10 )
            fprintf(' Warning TX\n');
        elseif ( iprint >= 10 )
            fprintf(' Warning: Change in x was less than options.TolX\n')
        end
        case 3
        if ( iprint == 1 )
            fprintf('\n');
        elseif ( iprint >= 2 && iprint < 10 )
            fprintf(' Warning TJ\n');
        elseif ( iprint >= 10 )
            fprintf([' Warning: Change in the objective function',...
                     ' value was less than options.TolFun\n'])
        end
        case 4
        if ( iprint == 1 )
            fprintf('\n');
        elseif ( iprint >= 2 && iprint < 10 )
            fprintf(' Warning S\n');
        elseif ( iprint >= 10 )
            fprintf([' Warning: Magnitude of the search direction',...
                     ' was less than 2*options.TolX and constraint',...
                     ' violation was less than options.TolCon\n'])
        end
        case 5
        if ( iprint == 1 )
            fprintf('\n');
        elseif ( iprint >= 2 && iprint < 10 )
            fprintf(' Warning D\n');
        elseif ( iprint >= 10 )
            fprintf([' Warning: Magnitude of directional derivative',...
                     ' in search direction was less than',...
                     ' 2*options.TolFun and maximum constraint',...
                     ' violation was less than options.TolCon\n'])
        end
    end
    if ( iprint >= 5 )
        plotTrajectories(@dynamic, system, T, t0, x0, u, atol_ode, rtol_ode, type)
    end
end

function printHeaderDummy(varargin)
end

function printClosedloopDataDummy(varargin)
end

function plotTrajectoriesDummy(varargin)
end
