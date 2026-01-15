close all;

%% example of drifting clocks compared to ideal clock
T=0:0.1:10;
Tlocal=0*abs(rand(size(T)))*0.8+T*0.9-0.005*T.*T-0.1;
Tremote=0*abs(rand(size(T)))*0.9+T*1.1;
figure(1);
hold on;
plot(T,T,'-k','LineWidth',2);
plot(T,Tlocal,'-b','LineWidth',2);
plot(T,Tremote,'-r','LineWidth',2);
hold off;
legend('Real time','Local time','Remote time');
xlabel('time');
ylabel('measured time');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

%% example of jumping offset
T1=[0,1,2,3,6,11];
Off=[0,1,-2,-1,-2,-2];
figure();
stairs(T1,Off,'-.ob');
axis([0 10 -3 3]);
xlabel('time');
ylabel('clock offset');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

% offset applied to function:
T2=0:0.01:10;
Off2=interp1(T1,Off,T2,'previous');
Discontinuities=find(diff(Off2));
Off2(Discontinuities)=NaN;
figure();
plot(T2,Off2+T2,'-','LineWidth',2);
axis([0 10 -1 9]);
xlabel('time');
ylabel('adjusted time');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

%% smoothed offset (computed manually from previous data)
T3=[0,1,2,5,6,8,12];
Off3=[0,0,0.5,-1,-1,-2,-2];

% supersample Off3
T4=0:0.1:10;
Off4=interp1(T3,Off3,T4);
figure();
stairs(T1,Off,'-.ob');
hold on;
plot(T4,Off4,'-m','LineWidth',2);
hold off;
axis([0 10 -3 1.5]);
xlabel('time');
ylabel('clock offset');
legend('offset','smoothed offset');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

figure();
plot(T4,Off4+T4,'LineWidth',2);
axis([0 10 -1 9]);
xlabel('time');
ylabel('adjusted time');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
