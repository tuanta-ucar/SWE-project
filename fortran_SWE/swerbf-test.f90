!
! ifort swerbf.f90 
!

module phys
  real*8, parameter :: alpha = 0.0D0     ! Angle of rotation measured from the equator.
  real*8, parameter :: a = 6.37122D6     ! Mean radius of the earth (meters).
  real*8, parameter :: omega = 7.292D-5  ! Rotation rate of the earth (1/seconds).
  real*8, parameter :: g = 9.806160D0    ! Gravitational constant (m/s^2).
  real*8, parameter :: gh0 = g*5960.0D0  ! Initial condition for the geopotential field (m^2/s^2).
end module phys

module math
implicit none
  real*8, parameter :: pi = 3.14159265358979323846264338D0
  real*8, parameter :: rad2deg = 360.0D0/(2.0D0*pi)
end module math

module dims
implicit none
  integer, parameter :: Nnodes=163842
  integer, parameter :: Nvar=4
  integer, parameter :: Nnbr=31
end module dims

module derivs
  use dims
implicit none

  ! derivative parameters

  real*8  :: eps = 14.0D0     ! controls width of Gaussian RBF (function of resolution)
  integer :: order = 4        ! hyper viscosity order 
  integer :: dim = 2          ! dimension of stencil (on the sphere dim = 2)
  real*8  :: gamma = -6.4D-22 ! amount of hyperviscosity applied, multiplies Laplacian^order 

  real*8  :: DPx(Nnbr+1,Nnodes)
  real*8  :: DPy(Nnbr+1,Nnodes)
  real*8  :: DPz(Nnbr+1,Nnodes)
  real*8  :: Lmat(Nnbr+1,Nnodes)
  integer :: idx(Nnbr,Nnodes)

end module derivs

module coords
  use dims
implicit none
 
  ! ================================
  ! Cartesian coordinates
  ! ================================

  real*8 x(Nnodes)
  real*8 y(Nnodes)
  real*8 z(Nnodes)

  ! ================================
  ! Spherical Coordinates:
  !    Longitude (la) 
  !    Latitude (th)
  ! ================================

  real*8 r(Nnodes)
  real*8 la(Nnodes)
  real*8 th(Nnodes)

  ! ==================================
  ! Cartesian to spherical coordinates
  ! ==================================

  real*8 :: c2s_u(Nnodes,2)
  real*8 :: c2s_v(Nnodes,2)
  real*8 :: c2s_w(Nnodes,2)

  ! =========================================
  ! Spherical projection operators (3 vector)
  ! =========================================

  real*8 p_u(Nnodes,3)
  real*8 p_v(Nnodes,3)
  real*8 p_w(Nnodes,3)

  real*8 p_u_t(3,Nnodes)
  real*8 p_v_t(3,Nnodes)
  real*8 p_w_t(3,Nnodes)

end module coords

module times
implicit none
  real*8, parameter :: dt=90.0D0
  real*8, parameter :: ndays = 1
  real*8, parameter :: secpday = 86400.0D0
  integer, parameter :: nsteps = ndays*NINT(secpday/dt)
end module times

program swe
  use phys
  use dims
  use coords
  use derivs
  use times
implicit none

  real*8 fcor(Nnodes)

  ! =======================
  ! Initial state variables
  ! =======================

  real*8 gh(Nnodes)
  real*8 uc(Nnodes,3)
  real*8 ghm(Nnodes)
  real*8 gradghm(Nnodes,3)

  real*8 nodes(NNodes,3)

  real*8 H(Nnodes,Nvar)
  real*8 F(Nnodes,Nvar)
  real*8 K(Nnodes,Nvar)

  real*8 d1(Nvar,Nnodes)
  real*8 d2(Nvar,Nnodes)
  real*8 d3(Nvar,Nnodes)
  real*8 d4(Nvar,Nnodes)

  ! Transposed versions of H,F,K
  real*8 H_t(Nvar,Nnodes)
  real*8 F_t(Nvar,Nnodes)
  real*8 K_t(Nvar,Nnodes)

  ! Transposed versions of uc, gradghm
  real*8 uc_t(3,Nnodes)
  real*8 gradghm_t(3,Nnodes)

  real*8 ra
  real*8 sum1
  real*8 sum2
  real*8 sum3

  real*8 :: tstart,tstop,tps
  real*8 :: tstart0, tstop0, tps0 ! evalCartRhs function
  real*8 :: tps1 ! 1st loop in evalCartRhs function

  integer :: nt
  integer :: i, inbr

  integer :: j
  integer :: recNum=1    !used to read binary file
  real*8 :: tempNum      !used to store value read from binary file
  real*8 :: checkSum=0.0D0 

  !INTERFACE 
  !  subroutine deriv_init(nodes)
  !     use dims
  !  implicit none
  !     real*8, intent(in) :: nodes(Nnodes,3)
  !  end subroutine deriv_init
  !END INTERFACE

  open(UNIT=7,FILE="icos163842",STATUS="OLD")
  do i=1,Nnodes
     read(7,*)nodes(i,1),nodes(i,2),nodes(i,3)
  end do
  close(unit=7)

  call coord_init(nodes)

  call deriv_init(nodes)

  call tc5(ghm,gradghm,gh,uc)

  H(:,1:3)=uc(:,1:3)
  H(:,4) = gh(:)
  fcor(:) = 2.0D0*omega*(-x(:)*sin(alpha) + z(:)*cos(alpha))   ! Coriolis force

  nt=0

  !=======================
  ! Optimization: Transpose H, gradghm to get more efficient memory access
  H_t = transpose(H)
  gradghm_t = transpose(gradghm)
  uc_t = transpose(uc)
  !=======================

  ! Initialize timing variables
  tps0=0.0D0
  tps1=0.0D0

  call cpu_time(tstart)
  do  nt=1,100!nsteps

      ! 
      ! Fourth Order Runge-Kutta timestepping
      !

      K_t = H_t
      
      call cpu_time(tstart0)
      call evalCartRhs(fcor,ghm,gradghm_t,K_t,F_t, tps1)
      call cpu_time(tstop0)
      tps0=tps0+(tstop0-tstart0)
      
      d1=dt*F_t
      K_t = H_t + 0.5D0*d1

      call cpu_time(tstart0)
      call evalCartRhs(fcor,ghm,gradghm_t,K_t,F_t, tps1)
      call cpu_time(tstop0)
      tps0=tps0+(tstop0-tstart0)

      d2 = dt*F_t
      K_t = H_t + 0.5D0*d2
      
      call cpu_time(tstart0)
      call evalCartRhs(fcor,ghm,gradghm_t,K_t,F_t, tps1)
      call cpu_time(tstop0)
      tps0=tps0+(tstop0-tstart0)

      d3 = dt*F_t
      K_t = H_t + d3
      
      call cpu_time(tstart0)
      call evalCartRhs(fcor,ghm,gradghm_t,K_t,F_t, tps1)
      call cpu_time(tstop0)
      tps0=tps0+(tstop0-tstart0)
      
      d4 = dt*F_t
      H_t = H_t + (1.0D0/6.0D0)*(d1 + 2.0D0*d2 + 2.0D0*d3 + d4)

  end do
  call cpu_time(tstop)

  !==================
  ! transpose H_t back to H for verification purpose
  H = transpose(H_t)
  !==================

  ! per-step time
  tps = (tstop-tstart)/100
  tps0 = tps0/100
  tps1 = tps1/100

  print *, "Time for the fused loop ", tps1
  print *, "Time for evalCartRhs ", tps0 
  print *, "Total (per step) ", tps
  
  ! Verify output by checking H matrix
  open(UNIT=22, FILE="H_debug.bin", STATUS="OLD", &
          FORM="UNFORMATTED", ACCESS="DIRECT", RECL=8)

  recNum=1
  
  do i=1,Nnodes
     do j=1, Nvar
       read(22, rec=recNum) tempNum

       if (ABS(H(i,j) - tempNum) > 1D-10) then
         print *, H(i,j), tempNum
       end if
 
       recNum = recNum + 1
     end do
  end do

  close(UNIT=22)

  tps=(tstop-tstart)/nsteps
  print *,(Nnodes*(4.0D0*(8.0D0*Nnbr*Nvar+64.0D0)+Nvar*16.0D0)*1.0D-9)/tps

end program swe

subroutine coord_init(nodes)
  use dims
  use coords
implicit none
  real*8, intent(in) :: nodes(Nnodes,3)

  x = nodes(:,1) 
  y = nodes(:,2) 
  z = nodes(:,3)

  call cart2sph(Nnodes,x,y,z,r,la,th)

  ! Variables for projecting an arbitrary Cartesian vector onto the surface
  ! of the sphere.

  p_u(:,1) = 1.0D0-x*x  
  p_u(:,2) = -x*y   
  p_u(:,3) = -x*z

  p_v(:,1) = -x*y
  p_v(:,2) = 1.0D0-y*y
  p_v(:,3) = -y*z

  p_w(:,1) = -x*z
  p_w(:,2) = -y*z
  p_w(:,3) = 1.0D0-z*z

  !=========================
  ! Optimization: transpose p_u, p_v and p_w for better memory access
  p_u_t = transpose(p_u)
  p_v_t = transpose(p_v)
  p_w_t = transpose(p_w)
  !=========================
  ! vectors for translating the field in Cartesian coordinates to a field
  ! in spherical coordinates.

  c2s_u(:,1) = -sin(la(:)) 
  c2s_u(:,2) = -sin(th(:))*cos(la(:))

  c2s_v(:,1) =  cos(la(:))  
  c2s_v(:,2) = -sin(th(:))*sin(la(:))

  c2s_w(:,1) =  0.0D0
  c2s_w(:,2) =  cos(th(:))

end subroutine coord_init

subroutine tc5(ghm,gradghm,gh,uc)
  use dims
  use math
  use phys
  use coords
  use derivs
implicit none

  real*8, intent(out)  :: ghm(Nnodes)
  real*8, intent(out)  :: gradghm(Nnodes,3)
  real*8, intent(out)  :: gh(Nnodes)
  real*8, intent(out)  :: uc(Nnodes,3)

  ! Parameters for the test case 5 (flow over mountain)

  real*8, parameter :: u0=20.0D0          ! wind speed
  real*8, parameter :: lam_c = 1.50D0*pi  ! latitude coord of mountain
  real*8, parameter :: thm_c = pi/6.0D0   ! longitude coord of mountain
  real*8, parameter :: mR = pi/9.0D0      ! length scale (radius) of mountain
  real*8, parameter :: hm0 = 2000.0D0     ! height of mountain

  real*8 :: sum1,sum2,sum3
  real*8 :: r2(Nnodes)

  integer :: i, inbr

  ! Compute the profile of the mountain (multiplied by gravity)

  ghm(:) = 0.0D0
  r2(:) = (la(:)-lam_c)**2 + (th(:)-thm_c)**2
  where(r2(:) < mR**2)
    ghm(:) = g*hm0*(1.0D0-sqrt(r2(:))/mR)
  end where

  ! ====================================
  ! Compute gradient of topography
  ! Assumes deriv_init has been called
  ! ====================================

  do i=1,Nnodes
     sum1=0.0D0
     sum2=0.0D0
     sum3=0.0D0
     do inbr=1,Nnbr
        sum1 = sum1+DPx(inbr,i)*ghm(idx(inbr,i))
        sum2 = sum2+DPy(inbr,i)*ghm(idx(inbr,i))
        sum3 = sum3+DPz(inbr,i)*ghm(idx(inbr,i))
     end do
     gradghm(i,1)=sum1
     gradghm(i,2)=sum2
     gradghm(i,3)=sum3
  end do

  ! Initial condition for test case 5.  uc contains the velocity in Cartesian 
  ! coordinates with uc(:,1:3) the x,y,z direction, respectively.  gh contains
  ! the geopotential height field from the mean geopotential gh0.

  gh(:) = - (a*omega*u0+ u0**2/2.0D0)*(-x(:)*sin(alpha) + z(:)*cos(alpha))**2
  uc(:,1) = -u0*y(:)*cos(alpha)
  uc(:,2) =  u0*(x(:)*cos(alpha) + z(:)*sin(alpha))
  uc(:,3) = -u0*y(:)*sin(alpha)

end subroutine tc5

subroutine evalCartRhs(fcor,ghm,gradghm_t,H_t,F_t, tps1)
  use phys
  use dims
  use derivs
  use coords
implicit none

real*8, intent(in)  ::  fcor(Nnodes)
real*8, intent(in)  ::  ghm(NNodes)
real*8, intent(in)  ::  gradghm_t(3,NNodes)
real*8, intent(in)  ::  H_t(Nvar,Nnodes)
real*8, intent(out) ::  F_t(Nvar,Nnodes)

real*8, intent(out) ::  tps1   ! timing variable for 1st loop

integer :: i, inbr, ivar

real*8 p,q,s

real*8 tstart, tstop ! timing variables

! temporary local variables
real*8 H_i1, H_i2, H_i3, H_i4
real*8 Tx_i1, Tx_i2, Tx_i3, Tx_i4
real*8 Ty_i1, Ty_i2, Ty_i3, Ty_i4
real*8 Tz_i1, Tz_i2, Tz_i3, Tz_i4
real*8 HV_i1, HV_i2, HV_i3, HV_i4
real*8 nbr_id

real*8 dp_x, dp_y, dp_z, l_mat

!
! Compute the (projected) Cartesian derivatives applied to the velocity
! and geopotential.
!

call cpu_time(tstart)
do i=1,Nnodes   ! 1st loop to be optimized

   ! 
   ! FLOPS1 = 8*Nnbr*Nvar (assumes precalculate DP{x,y,z}/a)
   !
   !

   Tx_i1 = 0.0D0
   Tx_i2 = 0.0D0
   Tx_i3 = 0.0D0
   Tx_i4 = 0.0D0
   
   Ty_i1 = 0.0D0
   Ty_i2 = 0.0D0
   Ty_i3 = 0.0D0
   Ty_i4 = 0.0D0
   
   Tz_i1 = 0.0D0
   Tz_i2 = 0.0D0
   Tz_i3 = 0.0D0
   Tz_i4 = 0.0D0
   
   HV_i1 = 0.0D0
   HV_i2 = 0.0D0
   HV_i3 = 0.0D0
   HV_i4 = 0.0D0

   do inbr=1,Nnbr
      nbr_id = idx(inbr,i)
      dp_x = DPx(inbr,i)
      dp_y = DPy(inbr,i)
      dp_z = DPz(inbr,i)
      l_mat = Lmat(inbr,i)
 
      Tx_i1 = Tx_i1 + dp_x*H_t(1,nbr_id)
      Ty_i1 = Ty_i1 + dp_y*H_t(1,nbr_id)
      Tz_i1 = Tz_i1 + dp_z*H_t(1,nbr_id)
      HV_i1 = HV_i1 + l_mat*H_t(1,nbr_id)
      
      Tx_i2 = Tx_i2 + dp_x*H_t(2,nbr_id)
      Ty_i2 = Ty_i2 + dp_y*H_t(2,nbr_id)
      Tz_i2 = Tz_i2 + dp_z*H_t(2,nbr_id)
      HV_i2 = HV_i2 + l_mat*H_t(2,nbr_id)
      
      Tx_i3 = Tx_i3 + dp_x*H_t(3,nbr_id)
      Ty_i3 = Ty_i3 + dp_y*H_t(3,nbr_id)
      Tz_i3 = Tz_i3 + dp_z*H_t(3,nbr_id)
      HV_i3 = HV_i3 + l_mat*H_t(3,nbr_id)

      Tx_i4 = Tx_i4 + dp_x*H_t(4,nbr_id)
      Ty_i4 = Ty_i4 + dp_y*H_t(4,nbr_id)
      Tz_i4 = Tz_i4 + dp_z*H_t(4,nbr_id)
      HV_i4 = HV_i4 + l_mat*H_t(4,nbr_id)
   end do

   !
   ! This is the computation for the right hand side of the (Cartesian) 
   ! momentum equations.
   !

   !
   ! FLOPS2 = 3*11
   !     
   ! store data used more than once in temporary variables
   
   H_i1 = H_t(1,i)
   H_i2 = H_t(2,i)
   H_i3 = H_t(3,i)
   H_i4 = H_t(4,i)

   p = -(H_i1*Tx_i1 + H_i2*Ty_i1 + H_i3*Tz_i1 + (fcor(i))*(y(i)*H_i3 - z(i)*H_i2) + Tx_i4)
   q = -(H_i1*Tx_i2 + H_i2*Ty_i2 + H_i3*Tz_i2 + (fcor(i))*(z(i)*H_i1 - x(i)*H_i3) + Ty_i4)
   s = -(H_i1*Tx_i3 + H_i2*Ty_i3 + H_i3*Tz_i3 + (fcor(i))*(x(i)*H_i2 - y(i)*H_i1) + Tz_i4)

   ! Project the momentum equations onto the surface of the sphere

   !
   ! FLOPS3 = 3*6
   !

   F_t(1,i) = p_u_t(1,i)*p + p_u_t(2,i)*q + p_u_t(3,i)*s+HV_i1
   F_t(2,i) = p_v_t(1,i)*p + p_v_t(2,i)*q + p_v_t(3,i)*s+HV_i2
   F_t(3,i) = p_w_t(1,i)*p + p_w_t(2,i)*q + p_w_t(3,i)*s+HV_i3

   ! Right-hand side for the geopotential (Does not need to be projected, this
   ! has already been accounted for in the DPx, DPy, and DPz operators for
   ! this equation).

   !
   ! FLOPS4 = 15
   !

   F_t(4,i) = -(H_i1*(Tx_i4 - gradghm_t(1,i)) + H_i2*(Ty_i4 - gradghm_t(2,i)) +  &
              H_i3*(Tz_i4 - gradghm_t(3,i)) + (H_i4+gh0-ghm(i))*(Tx_i1 + Ty_i2 + Tz_i3))+HV_i4

end do
call cpu_time(tstop)
tps1 = tps1 + (tstop-tstart)

! FLOPS = Nnodes*(8*Nnbr*Nvar+64)

end subroutine evalCartRhs

subroutine cart2sph(N,x,y,z,r,la,th)
  use math
implicit none
  integer, intent(in) :: N

  real*8, intent(in)  :: x(N)
  real*8, intent(in)  :: y(N)
  real*8, intent(in)  :: z(N)

  real*8, intent(out) :: r(N)
  real*8, intent(out) :: la(N)
  real*8, intent(out) :: th(N)

  real*8 :: s
  integer :: i

  r(:)  = SQRT(x*x+y*y+z*z)

  do i=1,N

     la(i) = atan2(y(i),x(i))
     if (y(i).lt.0.0D0) then
        la(i) = 2.0D0*pi+la(i)
     end if

     s= SQRT(x(i)*x(i)+y(i)*y(i))

     th(i) = atan2(z(i),s)

  end do

end subroutine cart2sph

subroutine deriv_init(nodes)
  use dims
  use derivs
  use phys 
  use derivs

implicit none
  real*8, intent(in) :: nodes(Nnodes,3)

  integer :: i
  integer :: j
  integer :: recNum=1 !record number

  open(UNIT=20, FILE="idx_transposed_binary.bin", STATUS="OLD", &
          FORM="UNFORMATTED", ACCESS="DIRECT", RECL=4)

  do i=1,Nnodes
     do j=1,Nnbr
       read(20, rec=recNum) idx(j,i)
       recNum = recNum + 1
     end do
  end do
  close(UNIT=20)

  open(UNIT=21, FILE="DP_binary.bin", STATUS="OLD", &
          FORM="UNFORMATTED", ACCESS="DIRECT", RECL=8)

  recNum=1
  ! read DPx
  do i=1,Nnodes
     do j=1, Nnbr
       read(21, rec=recNum) DPx(j,i)
       recNum = recNum + 1
     end do
  end do

  ! read DPy
  do i=1,Nnodes
     do j=1, Nnbr
       read(21, rec=recNum) DPy(j,i)
       recNum = recNum + 1
     end do
  end do
  
  ! read DPz   
  do i=1,Nnodes
     do j=1, Nnbr
       read(21, rec=recNum) DPz(j,i)
       recNum = recNum + 1
     end do
  end do
  
  ! read Lmat
  do i=1,Nnodes
     do j=1, Nnbr
       read(21, rec=recNum) Lmat(j,i)
       recNum = recNum + 1
     end do
  end do

  ! divide DPx, DPy, DPz by a
  do i=1,Nnodes
     do j=1, Nnbr
       DPx(j,i) = DPx(j,i) / a
       DPy(j,i) = DPy(j,i) / a
       DPz(j,i) = DPz(j,i) / a
       Lmat(j,i) = Lmat(j,i) * gamma
     end do 
  end do

  close(UNIT=21)

end subroutine deriv_init
