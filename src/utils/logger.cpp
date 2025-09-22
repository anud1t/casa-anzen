/**
 * @file logger.cpp
 * @brief Logging utility implementation for Casa Anzen Security System
 * @author Casa Anzen Team
 */

#include "utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <filesystem>

namespace casa_anzen {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() 
    : log_level_(Level::INFO)
    , console_output_(true) {
}

Logger::~Logger() {
    if (log_file_ && log_file_->is_open()) {
        log_file_->close();
    }
}

void Logger::log(Level level, const std::string& message) {
    if (level < log_level_) {
        return;
    }

    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::string timestamp = getTimestamp();
    std::string level_str = levelToString(level);
    std::string log_entry = "[" + timestamp + "] [" + level_str + "] " + message;
    
    if (console_output_) {
        std::cout << log_entry << std::endl;
    }
    
    if (log_file_ && log_file_->is_open()) {
        *log_file_ << log_entry << std::endl;
        log_file_->flush();
    }
}

void Logger::debug(const std::string& message) {
    log(Level::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(Level::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(Level::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(Level::ERROR, message);
}

void Logger::setLogFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    // Create directory if it doesn't exist
    std::filesystem::path file_path(filename);
    std::filesystem::path dir_path = file_path.parent_path();
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }
    
    log_file_ = std::make_unique<std::ofstream>(filename, std::ios::app);
    if (!log_file_->is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
        log_file_.reset();
    }
}

std::string Logger::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

std::string Logger::levelToString(Level level) {
    switch (level) {
        case Level::DEBUG:   return "DEBUG";
        case Level::INFO:    return "INFO";
        case Level::WARNING: return "WARN";
        case Level::ERROR:   return "ERROR";
        default:             return "UNKNOWN";
    }
}

} // namespace casa_anzen
