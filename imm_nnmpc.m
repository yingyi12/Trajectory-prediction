T = 1;


global ubar;
global eta;



ubar          = 5.0;
eta           = 0.2;

mpciterations = 200;
N             = 10;
T             = 1;
tmeasure      = 0;
xmeasure      = [10, 1, 8, 0];
u0            = zeros(2,N);
tol_opt       = 1e-2;
opt_option    = 0;
iprint        = 11;
type          = 'difference equation';
atol_ode_real = 1e-12;
rtol_ode_real = 1e-12;
atol_ode_sim  = 1e-4;
rtol_ode_sim  = 1e-4;


[xx1,yy1] = part1;
[xx2,yy2] = part2;
[xx3,yy3] = part3;
[xx4,yy4] = part4;

[EVt, EVx, EVu,xest,yest,xxpossible1,yypossible1,xxpossible2,yypossible2,xxpossible3,yypossible3,xp,ep,m,n,EVxpre,EVypre] = nnmpc(@runningcosts, @terminalcosts, @constraints, ...
    @terminalconstraints, @linearconstraints, @system_dt, ...
    mpciterations, N, T, tmeasure, xmeasure,u0, ...
    tol_opt, opt_option, ...
    type, atol_ode_real, rtol_ode_real, atol_ode_sim, rtol_ode_sim, ...
    iprint, @printHeader, @printClosedloopData, @plotTrajectories);
egox = EVx(:,1);
egoy = EVx(:,3);






f = figure(1);
v = VideoWriter('peaks.avi');
open(v);
frames = struct('cdata',zeros(600,600,3,'uint8'),'colormap',[]); % structure to save frames (substitute 600 and 600 with size of your image, if different)
plot([-20 500],[5 5],'-k','LineWidth',2);
hold on;
plot([-20 500],[15 15],'-k','LineWidth',2);
hold on;
plot([-20 500],[10 10],'--k','linewidth',1);
hold on;
for k=1:mpciterations

TV1 = rectangle('Position', [xest(1,k)-1 yest(1,k)-1.5 2 3], 'EdgeColor', 'b', 'LineWidth', 2);
hold on;
TV2 =rectangle('Position', [xest(2,k)-1 yest(2,k)-1.5 2 3], 'EdgeColor', 'b', 'LineWidth', 2);
axis equal;
hold on;
TV3 =rectangle('Position', [xest(3,k)-1 yest(3,k)-1 2 2], 'EdgeColor', 'b', 'LineWidth', 2);
hold on;
TV4 =rectangle('Position', [xest(4,k)-1 yest(4,k)-1 2 2], 'EdgeColor', 'b', 'LineWidth', 2);
hold on;

EV =rectangle('Position', [egox(k)-1 egoy(k)-1.5  2 3], 'EdgeColor', 'r', 'LineWidth', 2);
hold on;
p= plot(xx1(k),yy1(k),'r*',xxpossible1{1,k},yypossible1{1,k},'m*',xxpossible2{1,k},yypossible2{1,k},'c*', xxpossible3{1,k}, yypossible3{1,k},'g*', ...
        xx2(k),yy2(k),'r*',xxpossible1{2,k},yypossible1{2,k},'m*',xxpossible2{2,k},yypossible2{2,k},'c*', xxpossible3{2,k}, yypossible3{2,k},'g*', ...
        xx3(k),yy3(k),'r*',xxpossible1{3,k},yypossible1{3,k},'m*',xxpossible2{3,k},yypossible2{3,k},'c*', xxpossible3{3,k}, yypossible3{3,k},'g*', ...
        xx4(k),yy4(k),'r*',xxpossible1{4,k},yypossible1{4,k},'m*',xxpossible2{4,k},yypossible2{4,k},'c*', xxpossible3{4,k}, yypossible3{4,k},'g*', ...
        EVxpre(k,:),EVypre(k,:),'rx', ...
        ep{1,k},ep{2,k},'m-',ep{3,k},ep{4,k},'m-',ep{5,k},ep{6,k},'m-',ep{7,k},ep{8,k},'m-', ...
        ep{9,k},ep{10,k},'c-', ep{11,k},ep{12,k},'c-',ep{13,k},ep{14,k},'c-',ep{15,k},ep{16,k},'c-', ...
        ep{17,k},ep{18,k},'g-',ep{19,k},ep{20,k},'g-',ep{21,k},ep{22,k},'g-', ep{23,k},ep{24,k},'g-'); % plot real positions, predictions and so...
 %axis([10, 200, -90, 100])

 axis([egox(k)-30, egox(k)+30, egoy(k)-30, egoy(k)+30])


    
frames(k) = getframe(f); % save images as it appears, when you plot everything you need

delete(TV1)
delete(TV2)
delete(TV3)
delete(TV4)
delete(EV)

delete(p) % remove plots of current positions/predictions
writeVideo(v,frames(k));
end
close(f)
close(v);
m1 = implay(frames); % stores the frames in a video



function [t0, x0] = measureInitialValue ( tmeasure, xmeasure )
    t0 = tmeasure;
    x0 = xmeasure;
end

function cost = runningcosts(t, x, u)
    cost =0.25*(x(2)-5)^2+x(4)^2+0.2*(x(3)-8)^2;
end

function cost = terminalcosts(t, x)
    cost = 0.0;
end

function [c,ceq] = constraints(t, x, u,xp,m,n)
    global eta;
    c(1)  =   x(:,3)-15;
    c(2)  =  -x(:,3)+5; %stay on road
    c(3)  =   x(:,2)-10;
    c(4)  =  -x(:,2);
    c(5)  =   x(:,4)-1;
    c(6)  =  -x(:,4)-1;
    [Row, ~]=size(x);
    if Row<11
    c(7)  =  1-(x(:,1)-xp(:,1))^2./m(:,1)^2-(x(:,3)-xp(:,13))^2./n(:,1)^2;% TV1
    c(8)  =  1-(x(:,1)-xp(:,2))^2./m(:,2)^2-(x(:,3)-xp(:,14))^2./n(:,2)^2;% TV2
    c(9)  =  1-(x(:,1)-xp(:,3))^2./m(:,3)^2-(x(:,3)-xp(:,15))^2./n(:,3)^2;% TV3
    c(10) =  1-(x(:,1)-xp(:,4))^2./m(:,4)^2-(x(:,3)-xp(:,16))^2./n(:,4)^2;% TV4
    c(11) =  1-(x(:,1)-xp(:,5))^2./m(:,5)^2-(x(:,3)-xp(:,17))^2./n(:,5)^2;% TV1
    c(12) =  1-(x(:,1)-xp(:,6))^2./m(:,6)^2-(x(:,3)-xp(:,18))^2./n(:,6)^2;% TV2
    c(13) =  1-(x(:,1)-xp(:,7))^2./m(:,7)^2-(x(:,3)-xp(:,19))^2./n(:,7)^2;% TV3
    c(14) =  1-(x(:,1)-xp(:,8))^2./m(:,8)^2-(x(:,3)-xp(:,20))^2./n(:,8)^2;% TV4
    c(15) =  1-(x(:,1)-xp(:,9))^2./m(:,9)^2-(x(:,3)-xp(:,21))^2./n(:,9)^2;% TV1
    c(16) =  1-(x(:,1)-xp(:,10))^2./m(:,10)^2-(x(:,3)-xp(:,22))^2./n(:,10)^2;% TV2
    c(17) =  1-(x(:,1)-xp(:,11))^2./m(:,11)^2-(x(:,3)-xp(:,23))^2./n(:,11)^2;% TV3
    c(18) =  1-(x(:,1)-xp(:,12))^2./m(:,12)^2-(x(:,3)-xp(:,24))^2./n(:,12)^2;% TV4  
    else
    end


     

    ceq   =  [];
  
end

function [c,ceq] = terminalconstraints(t, x, xp,m,n)
    global eta;
    c(1)  =   x(:,3)-15;
    c(2)  =  -x(:,3)+5; %stay on road
    c(3)  =   x(:,2)-10;
    c(4)  =  -x(:,2);
    c(5)  =   x(:,4)-1;
    c(6)  =  -x(:,4)-1;
    [Row, ~]=size(x);
    if Row<11
    c(7)  =  1-(x(:,1)-xp(:,1))^2./m(:,1)^2-(x(:,3)-xp(:,13))^2./n(:,1)^2;% TV1
    c(8)  =  1-(x(:,1)-xp(:,2))^2./m(:,2)^2-(x(:,3)-xp(:,14))^2./n(:,2)^2;% TV2
    c(9)  =  1-(x(:,1)-xp(:,3))^2./m(:,3)^2-(x(:,3)-xp(:,15))^2./n(:,3)^2;% TV3
    c(10) =  1-(x(:,1)-xp(:,4))^2./m(:,4)^2-(x(:,3)-xp(:,16))^2./n(:,4)^2;% TV4
    c(11) =  1-(x(:,1)-xp(:,5))^2./m(:,5)^2-(x(:,3)-xp(:,17))^2./n(:,5)^2;% TV1
    c(12) =  1-(x(:,1)-xp(:,6))^2./m(:,6)^2-(x(:,3)-xp(:,18))^2./n(:,6)^2;% TV2
    c(13) =  1-(x(:,1)-xp(:,7))^2./m(:,7)^2-(x(:,3)-xp(:,19))^2./n(:,7)^2;% TV3
    c(14) =  1-(x(:,1)-xp(:,8))^2./m(:,8)^2-(x(:,3)-xp(:,20))^2./n(:,8)^2;% TV4
    c(15) =  1-(x(:,1)-xp(:,9))^2./m(:,9)^2-(x(:,3)-xp(:,21))^2./n(:,9)^2;% TV1
    c(16) =  1-(x(:,1)-xp(:,10))^2./m(:,10)^2-(x(:,3)-xp(:,22))^2./n(:,10)^2;% TV2
    c(17) =  1-(x(:,1)-xp(:,11))^2./m(:,11)^2-(x(:,3)-xp(:,23))^2./n(:,11)^2;% TV3
    c(18) =  1-(x(:,1)-xp(:,12))^2./m(:,12)^2-(x(:,3)-xp(:,24))^2./n(:,12)^2;% TV4  
    else
    end

    ceq   =  [];
end

function [A, b, Aeq, beq, lb, ub] = linearconstraints(t, x, u)
    global ubar
    A   = [];
    b   = [];
    Aeq = [];
    beq = [];
    lb  =  -5;
    ub  =  5;
end

function y = system_dt(t, x, u, T)
    y(1) = x(1)+T*x(2)+0.5*T^2*u(1);
    y(2) = x(2)+T*u(1);
    y(3) = x(3)+T*x(4)+0.5*T^2*u(2);
    y(4) = x(4)+T*u(2);
end

function dx = system_ct(t, x, u, T)
%     dx = zeros(4,1);
    dx(1) = x(2);
    dx(2) = u(1);
    dx(3) = x(4);
    dx(4) = u(2);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of output format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function printHeader()
    fprintf('   k  |    x(1)  x(2)  x(3)  x(4)   u(1)  u(2) Time\n');
    fprintf('--------------------------------------------------\n');
end

function printClosedloopData(mpciter, u, x, t_Elapsed)
           fprintf(' %3d | %+11.6f %+11.6f %+11.6f %+11.6f %+11.6f %+11.6f  %+6.3f', mpciter, ...
        x(1),x(2),x(3),x(4),u(1),u(2),t_Elapsed);
end

function plotTrajectories(dynamic, system, T, t0, x0, u, ...
                          atol_ode, rtol_ode, type)
    global eta;
    [x, t_intermediate, x_intermediate] = dynamic(system, T, t0, ...
                                          x0, u, atol_ode, rtol_ode, type);

end

