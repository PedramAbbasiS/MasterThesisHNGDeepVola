function F = forcefun(x,forces,sigma,derivative)
[n,~] = size(forces);
F = 0;
for i=1:n
    F = F-10.*sigma(i).*norm(x-forces(i,:)).^4.*exp(-10.*norm(x-forces(i,:)).^4).*2.*(x-forces(i,:)');
end


% angenommen das hier ist jetzt der main file

forces = ....
sigma = ....
derivative =....
Fun = @(x) forcefun(x,forces,sigma,derivatives)