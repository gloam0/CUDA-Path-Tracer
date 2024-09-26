#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <thread>

////////////////////////////////////////////////////////////////////////////////////////////////
using nanoseconds = std::chrono::nanoseconds;
using microseconds = std::chrono::microseconds;
using milliseconds = std::chrono::milliseconds;
using seconds = std::chrono::seconds;
using minutes = std::chrono::minutes;
using hours = std::chrono::hours;
////////////////////////////////////////////////////////////////////////////////////////////////
/// A timer for generic timing, frame timing, FPS tracking, and FPS limiting.
class Timer {
public:
    ////////////////////////////////////////////////////////////////////////////////////////
    Timer() : target_fps(60.) { init(); }
    ////////////////////////////////////////////////////////////////////////////////////////
    /// Start or restart the timer
    void start() { init(); }
    /// Stop the running timer
    void stop() {
        if (running) {
            stop_time = std::chrono::steady_clock::now();
            running = false;
        } else {
            std::cerr << "Timer: tried to stop() a timer which was not running.";
        }
    }
    /// Resume the stopped timer
    void resume() {
        if (!running) {
            auto const now = std::chrono::steady_clock::now();
            auto const stop_duration = now - stop_time;
            total_paused_duration += stop_duration;
            pause_duration_current_lap += stop_duration;
            start_time = now;
            running = true;
        } else {
            std::cerr << "Timer: tried to resume() a timer which was already running.";
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    /// Elapsed time since last usage of lap().
    template<class T>
    double lap() {
        auto const now = running ? std::chrono::steady_clock::now() : stop_time;
        auto const duration = now - last_lap_time_point - pause_duration_current_lap;
        pause_duration_current_lap = std::chrono::steady_clock::duration::zero();
        last_lap_time_point = now;
        return std::chrono::duration_cast<T>(duration).count();
    }
    // Elapsed time since instantiation or start().
    template<class T>
    double duration() {
        auto const now = running ? std::chrono::steady_clock::now() : stop_time;
        auto const duration = now - start_time - total_paused_duration;
        return std::chrono::duration_cast<T>(duration).count();
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    /// Set the FPS rate to adjust for
    void start_fps() {
        fps_start = std::chrono::steady_clock::now();
    }
    void set_target_fps(double fps) {
        target_fps = fps;
    }
    /// Called at the start of each frame. Limit FPS by waiting for the remainder of
    /// the current frame's expected duration at target_fps.
    void limit_fps() {
        if (!frame_count++) return;

        auto const now = std::chrono::steady_clock::now();
        auto const frame_duration = now - frame_start;
        auto const target_duration = std::chrono::duration<double>(1. / target_fps);

        if (frame_duration < target_duration) {
            std::this_thread::sleep_for(target_duration - frame_duration);
        }

        frame_start = std::chrono::steady_clock::now();
        frame_count++;
    }
    /// Should not be used where limit_fps() is used. Useful for keeping frame count
    /// accurate for get_current_fps() where limit_fps() is not already in use, and
    /// keeping frame_start accurate for get_frame_time().
    void frame_now() {
        frame_count++;
        frame_start = std::chrono::steady_clock::now();
    }
    /// Get FPS average over the most recently completed one-second interval
    double get_current_fps() {
        auto const now = std::chrono::steady_clock::now();
        auto const duration = now - fps_start;
        if (duration >= std::chrono::seconds(1)) {
            current_fps = static_cast<double>(frame_count)
                / std::chrono::duration<double>(duration).count();
            frame_count = 0;
            fps_start = now;
        }
        return current_fps;
    }
    /// Get the amount of time elapsed since frame_start
    double get_frame_time() const {
        auto frame_time = std::chrono::steady_clock::now() - frame_start;
        return std::chrono::duration_cast<microseconds>(frame_time).count();
    }
    ////////////////////////////////////////////////////////////////////////////////////////
private:
    std::chrono::steady_clock::time_point   start_time;
    std::chrono::steady_clock::time_point   stop_time;
    std::chrono::steady_clock::time_point   last_lap_time_point;
    std::chrono::steady_clock::duration     total_paused_duration;
    std::chrono::steady_clock::duration     pause_duration_current_lap;
    bool                                    running;

    std::chrono::steady_clock::time_point   frame_start;
    std::chrono::steady_clock::time_point   fps_start;
    uint64_t                                frame_count;
    double                                  target_fps;
    double                                  current_fps;

    void init() {
        auto const now = std::chrono::steady_clock::now();
        start_time = now;
        last_lap_time_point = now;
        frame_start = now;
        fps_start = now;
        total_paused_duration = std::chrono::steady_clock::duration::zero();
        pause_duration_current_lap = std::chrono::steady_clock::duration::zero();
        running = true;
        frame_count = 0;
        current_fps = 0.0;
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //TIMER_H
