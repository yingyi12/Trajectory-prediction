function [xx,yy]=part2
T =1;
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
x0 = [40;1;0;-17;1;0];

for k=1:200

time(k)=k*1;
    if k<50
        if mod(k,2) == 1
            x0 = F{1}*x0;
        else
            x0 = F{2}*x0;
        end
        
    else
        if k > 80
         x0 = F{3}*x0;
        else
            if mod(k,2) == 1
                x0 = F{3}*x0;
            else
                x0 = F{2}*x0;
            end

        end
       
        

    end
xxk(k,:)=x0;
xx = xxk(:,1);
yy = xxk(:,4);

end
%plot(xx(:),yy(:));
