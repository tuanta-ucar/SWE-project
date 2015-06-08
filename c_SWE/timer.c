#include <timer.h>

// Time measured in microseconds
long long getTime(){
        struct timeval time;

        int err = gettimeofday(&time, 0);

        return (long long)(time.tv_sec * 1000000 + time.tv_usec);
}

