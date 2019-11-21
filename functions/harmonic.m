
% 0,0    1/sqrt(4*pi)
% 1,0    z*sqrt(3/(4*pi))
% 1,1e    x*sqrt(3/(4*pi))
% 1,1o    y*sqrt(3/(4*pi))
% 2,0   (2*z.^2 - x.^2 - y.^2)/2 * sqrt(5/(4*pi))
% 2,1e  x*z * 3*sqrt(5/(12*pi))
% 2,1o  y*z * 3*sqrt(5/(12*pi))
% 2,2e  (x.^2-y.^2) * 3*sqrt(5/(48*pi))
% 2,2o  x*y * 3*sqrt(5/(12*pi))

xs = (N(1,:))'; ys = (N(2,:))'; zs = (N(3,:))';
a = sqrt(xs.^2+ys.^2+zs.^2);
%size(a) = 128*128 * 1
denom = (a==0) + a;
%x = xs./a; y = ys./a; z = zs./a;
x = xs ./ denom;
y = ys ./ denom;
z = zs ./ denom;

normal_im = [x,y,z];
normal_im = reshape(normal_im,[128,128,3]);
imshow(normal_im)
pause();
size(normal_im)

x2 = x.*x; y2 = y.*y; z2 = z.*z;
xy = x.*y; xz = x.*z; yz = y.*z;

H1 = att(1)*(1/sqrt(4*pi)) * a;
H2 = att(2)*(sqrt(3/(4*pi))) * zs;
H3 = att(2)*(sqrt(3/(4*pi))) * xs;
H4 = att(2)*(sqrt(3/(4*pi))) * ys;
H5 = att(3)*(1/2)*(sqrt(5/(4*pi))) * ((2*z2 - x2 - y2) .* a);
H6 = att(3)*(3*sqrt(5/(12*pi))) * (xz .* a);
H7 = att(3)*(3*sqrt(5/(12*pi))) * (yz .* a);
H8 = att(3)*(3*sqrt(5/(48*pi))) * ((x2 - y2) .* a);
H9 = att(3)*(3*sqrt(5/(12*pi))) *(xy .* a);
H = [H1,H2,H3,H4,H5,H6,H7,H8,H9];