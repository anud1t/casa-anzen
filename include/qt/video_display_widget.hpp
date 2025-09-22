#pragma once

/**
 * @file video_display_widget.hpp
 * @brief Video display widget with detection overlays for Casa Anzen
 * @author Casa Anzen Team
 */

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QDateTime>
#include <vector>
#include <opencv2/opencv.hpp>
#include "core/types.hpp"

namespace casa_anzen {

class VideoDisplayWidget : public QWidget {
    Q_OBJECT

public:
    explicit VideoDisplayWidget(QWidget *parent = nullptr);
    ~VideoDisplayWidget();

    // Video display methods
    void updateFrame(const cv::Mat& frame);
    void setOverlaysEnabled(bool enabled);
    void setDetectionOverlays(const std::vector<casa_anzen::TrackedObject>& tracks,
                             const std::vector<casa_anzen::Detection>& detections);
    void setAlertOverlays(const std::vector<casa_anzen::SecurityAlert>& alerts);
    void setZoneOverlays(const std::vector<casa_anzen::SecurityZone>& zones);

    // Drawing mode controls
    void setDrawModeEnabled(bool enabled) { m_draw_mode = enabled; update(); }
    bool isDrawModeEnabled() const { return m_draw_mode; }

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void drawDetections(QPainter& painter);
    void drawTracks(QPainter& painter);
    void drawAlerts(QPainter& painter);
    void drawZones(QPainter& painter);
    void drawSystemInfo(QPainter& painter);
    void drawBoundingBox(QPainter& painter, const QRect& rect, const QColor& color, int thickness);
    void drawLabel(QPainter& painter, const QString& label, const QColor& color, int x, int y);
    QColor getClassColor(const std::string& class_name) const;
    
    QRect cvRectToQRect(const cv::Rect& rect) const;
    QPoint cvPointToQPoint(const cv::Point2f& point) const;
    QRect normalizeRect(const QRect& rect) const; // ensure positive width/height
    cv::Rect mapDisplayRectToFrame(const QRect& displayRect) const;
    cv::Point mapDisplayPointToFrame(const QPoint& displayPoint) const;

    // Display data
    cv::Mat m_current_frame;
    std::vector<casa_anzen::TrackedObject> m_tracks;
    std::vector<casa_anzen::Detection> m_detections;
    std::vector<casa_anzen::SecurityAlert> m_alerts;
    std::vector<casa_anzen::SecurityZone> m_zones;
    
    // Display settings
    bool m_overlays_enabled;
    bool m_detection_overlays_enabled;
    bool m_track_overlays_enabled;
    bool m_alert_overlays_enabled;
    bool m_zone_overlays_enabled;
    
    // Scaling
    float m_scale_x;
    float m_scale_y;
    QSize m_display_size;

    // Draw mode state
    bool m_draw_mode { false };
    // Polygon drawing state
    bool m_poly_drawing { false };
    std::vector<QPoint> m_poly_points;
    QRect m_selected_rect;
    QDateTime m_selection_expires;

signals:
    void zoneCreated(const casa_anzen::SecurityZone& zone, const cv::Mat& frame);
    void captureRequested(const QString& class_name, const cv::Rect& bbox, const cv::Mat& frame);
};

} // namespace casa_anzen
