#pragma once

/**
 * @file logger.hpp
 * @brief Logging utility for Casa Anzen Security System
 * @author Casa Anzen Team
 */

#include <string>
#include <memory>
#include <fstream>
#include <mutex>

namespace casa_anzen {

class Logger {
public:
    enum class Level {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };

    // Singleton pattern
    static Logger& getInstance();
    
    // Disable copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Logging methods
    void log(Level level, const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

    // Configuration
    void setLogLevel(Level level) { log_level_ = level; }
    void setLogFile(const std::string& filename);
    void enableConsoleOutput(bool enable) { console_output_ = enable; }

private:
    Logger();
    ~Logger();

    Level log_level_;
    bool console_output_;
    std::unique_ptr<std::ofstream> log_file_;
    std::mutex log_mutex_;

    std::string getTimestamp();
    std::string levelToString(Level level);
};

} // namespace casa_anzen
