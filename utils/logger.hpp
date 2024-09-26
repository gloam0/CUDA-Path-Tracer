#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <ctime>

////////////////////////////////////////////////////////////////////////////////////////////////
/// Class for appending messages to a log file
class Logger {
public:
    enum TimeFormat { TIME_ONLY, WITH_DATE };    /* format for log timestamps */
    ////////////////////////////////////////////////////////////////////////////////////////
    explicit Logger(const std::filesystem::__cxx11::path& log_file_path)
        : log_file_path(log_file_path)
    {
        if (!open_log_file()) {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
            return;
        }

        log_file << "------------- " << get_local_time(WITH_DATE)
            << " Logger Start -------------\n" << std::endl;
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    ~Logger() {
        if (log_file.is_open()) {
            log_file << "\n------------- " << get_local_time(WITH_DATE)
                << " Logger End ---------------\n" << std::endl;
            log_file.close();
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    void log(const std::string& msg) {
        if (!open_log_file()) {
            std::cerr << "Failed to open log file: " << msg << std::endl;
            return;
        }

        log_file << get_local_time(TIME_ONLY) << msg << std::flush;

        if (log_file.fail()) {
            std::cerr << "Log write failed: " << msg << std::endl;
            log_file.close();
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    /// Enable stream operator
    template<typename T>
    Logger &operator<<(const T &data) {
        if (!open_log_file()) {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
            return *this;
        }

        log_file << data;
        return *this;
    }

    Logger &operator<<(std::ostream & (*manip)(std::ostream &)) {
        if (!open_log_file()) {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
            return *this;
        }

        log_file << manip;
        return *this;
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    /// @param time_format  (default: TIME_ONLY) include the date in the timestamp string
    /// @return             a string representing the current local timestamp
    static std::string get_local_time(TimeFormat time_format = TIME_ONLY) {
        auto now = std::chrono::system_clock::now();
        auto now_t = std::chrono::system_clock::to_time_t(now);

        std::ostringstream oss;
        oss << '[';
        if (time_format == WITH_DATE) {
            oss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S");
        } else {
            oss << std::put_time(std::localtime(&now_t), "%H:%M:%S");
        }
        oss << ']';
        return oss.str();
    }
    ////////////////////////////////////////////////////////////////////////////////////////
private:
    const std::filesystem::path  log_file_path;  /* path to log file */
    std::ofstream                log_file;       /* log file file stream */
    ////////////////////////////////////////////////////////////////////////////////////////
    /// Return true if log file is open, else attempt to open log file
    bool open_log_file() {
        if (!log_file.is_open())
            log_file.open(log_file_path, std::ios::app);

        return log_file.is_open();
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //LOGGER_H
