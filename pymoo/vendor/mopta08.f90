
program mopta08

    implicit      none

    character*200  in_file
    character*200  out_file
    integer        nvar
    integer        ncon
    parameter      (nvar = 124)
    parameter      (ncon =  68)
    integer        i
    real*8         x(nvar)
    real*8         f
    real*8         g(ncon)

    in_file  = "input.txt"
    out_file = "output.txt"

    do i=1,nvar
        read(*,*) x(i)
    enddo

    call func(nvar,ncon,x,f,g)

    write(*,'(F27.16)') f
    do i=1,ncon
        write(*,'(F27.16)') g(i)
    enddo

    stop

end