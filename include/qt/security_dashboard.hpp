#pragma once

/**
 * @file security_dashboard.hpp
 * @brief Main security dashboard window for Casa Anzen
 * @author Casa Anzen Team
 */

#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QStatusBar>
#include <QTimer>
#include <QDockWidget>
#include <QListWidget>
#include <QSplitter>
#include <QPushButton>
#include <QLineEdit>
#include <memory>
#include <set>

class QNetworkAccessManager;

// Include core types
#include "core/types.hpp"
#include "core/security_detector.hpp"
#include "core/zone_manager.hpp"
#include "core/recording_manager.hpp"

namespace casa_anzen {

// Forward declarations
class VideoDisplayWidget;
class VideoProcessingThread;
class QProgressBar;

/**
 * @brief Main security dashboard window for Casa Anzen Security System
 * 
 * This window provides a comprehensive security monitoring interface with:
 * - Live video feed with detection overlays
 * - Security zone management
 * - Alert monitoring and management
 * - Recording controls
 * - System status information
 */
class SecurityDashboard : public QMainWindow {
    Q_OBJECT

public:
    explicit SecurityDashboard(QWidget *parent = nullptr);
    ~SecurityDashboard();

    // Configuration methods
    void setModelPath(const std::string& model_path);
    void setVideoSource(const std::string& video_source);
    void setConfidenceThreshold(float threshold);
    void enableRecording(bool enable);
    void enableDebugMode(bool enable);
    void setRtspLatency(int latency_ms);
    void setRtspBufferSize(int buffer_size);
    void setRtspQuality(const std::string& quality);
    void autoStartIfConfigured();

public slots:
    void startProcessing();
    void stopProcessing();
    void openVideoFile();
    void openModelFile();
    void handleProcessingError(const QString& error_message);
    void updateStatus();
    void onNewFrame(const cv::Mat& frame);
    void updateDetectionData(const std::vector<casa_anzen::TrackedObject>& tracks,
                           const std::vector<casa_anzen::Detection>& detections);
    void updateSecurityAlerts(const std::vector<casa_anzen::SecurityAlert>& alerts);
    void toggleFullscreen();
    

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void setupUI();
    void setupConnections();
    void setupMenuBar();
    void createStatusBar();
    void updateSystemStatus();
    QWidget* createEventCard(const QString& title, const QPixmap& thumbnail, const QString& caption = QString());
    void captionItem(QListWidgetItem* item);
    void previewItem(QListWidgetItem* item);
    void deleteItem(QListWidgetItem* item);
    void deleteAllCaptures();

    // UI Components
    QWidget* m_central_widget;
    QVBoxLayout* m_main_layout;
    VideoDisplayWidget* m_video_display;
    QDockWidget* m_side_dock;
    QWidget* m_side_panel;
    QVBoxLayout* m_side_layout;
    QLabel* m_live_stats_header;
    QLabel* m_events_header;
    QListWidget* m_event_list;
    // Event toolbar
    QWidget* m_event_toolbar;
    QPushButton* m_view_btn;
    QPushButton* m_caption_btn;
    QPushButton* m_delete_btn;
    QPushButton* m_delete_all_btn;
    // Zone controls
    QLabel* m_zones_header;
    QWidget* m_zone_controls_box;
    QLineEdit* m_zone_name_edit;
    QPushButton* m_draw_btn;
    QPushButton* m_clear_zones_btn;
    QPushButton* m_toggle_zones_btn;
    bool m_zones_visible { true };
    QStatusBar* m_status_bar;
    QLabel* m_status_label;
    QLabel* m_fps_label;
    QLabel* m_detections_label;
    QLabel* m_alerts_label;
    QLabel* m_recording_label;
    QLabel* m_mode_label;
    
    // Networking
    QNetworkAccessManager* m_network { nullptr };
    
    // Processing thread
    std::unique_ptr<VideoProcessingThread> m_processing_thread;
    
    // Core components
    std::unique_ptr<casa_anzen::SecurityDetector> m_security_detector;
    std::unique_ptr<casa_anzen::ZoneManager> m_zone_manager;
    std::unique_ptr<casa_anzen::RecordingManager> m_recording_manager;
    
    // Configuration
    std::string m_model_path;
    std::string m_video_source;
    float m_confidence_threshold;
    bool m_recording_enabled;
    bool m_debug_mode;
    int m_rtsp_latency = 200;  // RTSP latency in milliseconds
    int m_rtsp_buffer_size = 10;  // RTSP buffer size
    std::string m_rtsp_quality = "high";  // RTSP quality mode
    int m_current_mode = 2; // 0=PEOPLE,1=VEHICLES,2=PEOPLE+VEHICLES,3=ALL
    
    // Status
    std::vector<casa_anzen::TrackedObject> m_current_tracks;
    std::vector<casa_anzen::Detection> m_current_detections;
    std::vector<casa_anzen::SecurityAlert> m_current_alerts;
    int m_frame_count;
    float m_current_fps;
    QTimer* m_update_timer;

    // Zones and entry tracking
    int m_zone_seq {0};
    std::set<std::string> m_inside_keys; // keys like "Zone 1:42"

    // Last frame for capture saving
    cv::Mat m_last_frame;
};

} // namespace casa_anzen
