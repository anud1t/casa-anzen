/**
 * @file video_display_widget.cpp
 * @brief Video display widget implementation for Casa Anzen
 * @author Anudit Gautam
 */

#include "qt/video_display_widget.hpp"
#include <QPainter>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QDateTime>
#include <QMouseEvent>

namespace casa_anzen {

VideoDisplayWidget::VideoDisplayWidget(QWidget *parent)
    : QWidget(parent)
    , m_overlays_enabled(true)
    , m_detection_overlays_enabled(true)
    , m_track_overlays_enabled(true)
    , m_alert_overlays_enabled(true)
    , m_zone_overlays_enabled(true)
    , m_scale_x(1.0f)
    , m_scale_y(1.0f) {
    
    setMinimumSize(640, 480);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

VideoDisplayWidget::~VideoDisplayWidget() {
}

void VideoDisplayWidget::updateFrame(const cv::Mat& frame) {
    m_current_frame = frame.clone();
    update(); // Trigger repaint
}

void VideoDisplayWidget::setOverlaysEnabled(bool enabled) {
    m_overlays_enabled = enabled;
    update();
}

void VideoDisplayWidget::setDetectionOverlays(const std::vector<casa_anzen::TrackedObject>& tracks,
                                             const std::vector<casa_anzen::Detection>& detections) {
    m_tracks = tracks;
    m_detections = detections;
    update();
}

void VideoDisplayWidget::setAlertOverlays(const std::vector<casa_anzen::SecurityAlert>& alerts) {
    m_alerts = alerts;
    update();
}

void VideoDisplayWidget::setZoneOverlays(const std::vector<casa_anzen::SecurityZone>& zones) {
    m_zones = zones;
    update();
}

void VideoDisplayWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    
    // Fill background
    painter.fillRect(rect(), Qt::black);
    
    if (m_current_frame.empty()) {
        // Draw placeholder text
        painter.setPen(Qt::white);
        painter.setFont(QFont("Arial", 16));
        painter.drawText(rect(), Qt::AlignCenter, "No video feed");
        return;
    }
    
    // Calculate scaling
    QSize frame_size(m_current_frame.cols, m_current_frame.rows);
    QSize widget_size = size();
    
    float scale_x = static_cast<float>(widget_size.width()) / frame_size.width();
    float scale_y = static_cast<float>(widget_size.height()) / frame_size.height();
    float scale = std::min(scale_x, scale_y);
    
    int scaled_width = static_cast<int>(frame_size.width() * scale);
    int scaled_height = static_cast<int>(frame_size.height() * scale);
    
    int x = (widget_size.width() - scaled_width) / 2;
    int y = (widget_size.height() - scaled_height) / 2;
    
    QRect draw_rect(x, y, scaled_width, scaled_height);
    
    // Convert OpenCV Mat to QImage (do NOT mutate m_current_frame to preserve BGR for saving)
    QImage qimg;
    if (m_current_frame.channels() == 3) {
        cv::Mat rgb;
        cv::cvtColor(m_current_frame, rgb, cv::COLOR_BGR2RGB);
        qimg = QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    } else if (m_current_frame.channels() == 1) {
        qimg = QImage(m_current_frame.data, m_current_frame.cols, m_current_frame.rows, 
                     m_current_frame.step, QImage::Format_Grayscale8).copy();
    }
    
    if (!qimg.isNull()) {
        // Draw video frame
        painter.drawImage(draw_rect, qimg.scaled(scaled_width, scaled_height, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        
        // Update scaling factors for overlays
        m_scale_x = static_cast<float>(scaled_width) / frame_size.width();
        m_scale_y = static_cast<float>(scaled_height) / frame_size.height();
        m_display_size = QSize(scaled_width, scaled_height);
        
        // Draw overlays if enabled
        if (m_overlays_enabled) {
            painter.setRenderHint(QPainter::Antialiasing);
            
            if (m_detection_overlays_enabled) {
                drawDetections(painter);
            }
            
            if (m_track_overlays_enabled) {
                drawTracks(painter);
            }
            
            if (m_alert_overlays_enabled) {
                drawAlerts(painter);
            }
            
            if (m_zone_overlays_enabled) {
                drawZones(painter);
            }
            
            drawSystemInfo(painter);
            // Draw in-progress polygon on top
            if (m_draw_mode && !m_poly_points.empty()) {
                painter.setPen(QPen(Qt::cyan, 2));
                for (int i = 0; i < static_cast<int>(m_poly_points.size()) - 1; ++i) {
                    painter.drawLine(m_poly_points[i], m_poly_points[i+1]);
                }
                // Draw points
                painter.setBrush(Qt::cyan);
                for (const auto& p : m_poly_points) {
                    painter.drawEllipse(p, 4, 4);
                }
            }
            // Draw temporary selection highlight
            if (!m_selected_rect.isNull() && QDateTime::currentDateTime() < m_selection_expires) {
                painter.setPen(QPen(Qt::yellow, 3));
                painter.drawRect(m_selected_rect);
            }
        }
    }
}

void VideoDisplayWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    update();
}

void VideoDisplayWidget::mousePressEvent(QMouseEvent* event) {
    if (!m_draw_mode) {
        // Click-to-capture on existing track bbox
        if (!m_current_frame.empty() && event->button() == Qt::LeftButton) {
            // Find the topmost track under cursor
            for (auto it = m_tracks.rbegin(); it != m_tracks.rend(); ++it) {
                const auto& t = *it;
                QRect rect = cvRectToQRect(cv::Rect(static_cast<int>(t.bbox.x),
                                                    static_cast<int>(t.bbox.y),
                                                    static_cast<int>(t.bbox.width),
                                                    static_cast<int>(t.bbox.height)));
                if (rect.contains(event->pos())) {
                    m_selected_rect = rect;
                    m_selection_expires = QDateTime::currentDateTime().addMSecs(600);
                    // Emit capture request with full bbox in frame coordinates
                    cv::Rect frameRect(static_cast<int>(t.bbox.x), static_cast<int>(t.bbox.y),
                                       static_cast<int>(t.bbox.width), static_cast<int>(t.bbox.height));
                    emit captureRequested(QString::fromStdString(t.class_name), frameRect, m_current_frame);
                    update();
                    return;
                }
            }
        }
        QWidget::mousePressEvent(event);
        return;
    }
    if (event->button() == Qt::LeftButton) {
        QPoint clickPos = event->pos();
        // Start polygon or add vertex
        if (!m_poly_drawing) {
            m_poly_drawing = true;
            m_poly_points.clear();
            m_poly_points.push_back(clickPos);
        } else {
            // If close to first point, close polygon
            if (!m_poly_points.empty() && (QLineF(clickPos, m_poly_points.front()).length() < 10.0)) {
                // Close polygon
                if (m_poly_points.size() >= 3 && !m_current_frame.empty()) {
                    // Map to frame coordinates and emit zone
                    std::vector<cv::Point> poly;
                    poly.reserve(m_poly_points.size());
                    for (const auto& p : m_poly_points) poly.push_back(mapDisplayPointToFrame(p));
                    casa_anzen::SecurityZone zone;
                    zone.name = "Zone_" + std::to_string(QDateTime::currentMSecsSinceEpoch());
                    zone.type = casa_anzen::ZoneType::MONITORED;
                    zone.enabled = true;
                    zone.alert_level = casa_anzen::SecuritySeverity::HIGH;
                    zone.description = "";
                    zone.recording_enabled = true;
                    zone.min_detection_time = 0;
                    zone.polygon = poly;
                    emit zoneCreated(zone, m_current_frame);
                }
                m_poly_points.clear();
                m_poly_drawing = false;
            } else {
                m_poly_points.push_back(clickPos);
            }
        }
        update();
    } else if (event->button() == Qt::RightButton) {
        // Cancel polygon drawing
        m_poly_points.clear();
        m_poly_drawing = false;
        update();
    }
}

void VideoDisplayWidget::mouseMoveEvent(QMouseEvent* event) {
    if (!m_draw_mode) { QWidget::mouseMoveEvent(event); return; }
    // Optionally show preview line to cursor
    if (m_poly_drawing && !m_poly_points.empty()) {
        // Draw a temporary segment to cursor by appending and removing last point
        // We only trigger repaint; actual drawing is handled in paintEvent using current points
        update();
    } else {
        QWidget::mouseMoveEvent(event);
    }
}

void VideoDisplayWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (!m_draw_mode) { QWidget::mouseReleaseEvent(event); return; }
    // No action on release for polygon drawing
    QWidget::mouseReleaseEvent(event);
}


void VideoDisplayWidget::drawDetections(QPainter& painter) {
    for (const auto& detection : m_detections) {
        QRect rect = cvRectToQRect(cv::Rect(static_cast<int>(detection.bbox.x), 
                                           static_cast<int>(detection.bbox.y),
                                           static_cast<int>(detection.bbox.width), 
                                           static_cast<int>(detection.bbox.height)));
        
        // Draw corner-style bounding box like helios-nano
        QColor color = getClassColor(detection.class_name);
        drawBoundingBox(painter, rect, color, 2);
        
        // Draw label with background (ALL CAPS class, no confidence)
        QString label = QString::fromStdString(detection.class_name).toUpper();
        // Center label above bounding box and raise position
        int label_x = rect.x() + (rect.width() - painter.fontMetrics().horizontalAdvance(label)) / 2;
        int label_y = rect.y() - 15; // Raised from -5 to -15
        drawLabel(painter, label, color, label_x, label_y);
    }
}

void VideoDisplayWidget::drawTracks(QPainter& painter) {
    for (const auto& track : m_tracks) {
        const cv::Rect2f& boxToDraw = (track.smoothed_bbox.width > 0 && track.smoothed_bbox.height > 0)
            ? track.smoothed_bbox
            : track.bbox;
        QRect rect = cvRectToQRect(cv::Rect(static_cast<int>(boxToDraw.x), 
                                           static_cast<int>(boxToDraw.y),
                                           static_cast<int>(boxToDraw.width), 
                                           static_cast<int>(boxToDraw.height)));
        
        // Choose color based on track status
        QColor color = getClassColor(track.class_name);
        if (track.is_lost) {
            color = Qt::gray;
        }
        
        // Draw corner-style bounding box like helios-nano
        drawBoundingBox(painter, rect, color, 2);

        // If user clicks inside this rect while not in draw mode, highlight and request capture
        // (handled in mousePressEvent override in the widget consumer, but we keep state here)
        
        // Draw class name only (ALL CAPS), no ID or confidence
        QString label = QString::fromStdString(track.class_name).toUpper();
        
        // Identity tags removed for cleaner display
        
        // Center label above bounding box and raise position
        int label_x = rect.x() + (rect.width() - painter.fontMetrics().horizontalAdvance(label)) / 2;
        int label_y = rect.y() - 15; // Raised from -5 to -15
        drawLabel(painter, label, color, label_x, label_y);
        
        // Draw trajectory
        if (track.trajectory.size() > 1) {
            painter.setPen(QPen(color, 1, Qt::DashLine));
            QPoint prev_point = cvPointToQPoint(track.trajectory[0]);
            for (size_t i = 1; i < track.trajectory.size(); i++) {
                QPoint current_point = cvPointToQPoint(track.trajectory[i]);
                painter.drawLine(prev_point, current_point);
                prev_point = current_point;
            }
        }
    }
}

void VideoDisplayWidget::drawAlerts(QPainter& painter) {
    for (const auto& alert : m_alerts) {
        if (alert.acknowledged) continue;
        
        QPoint center = cvPointToQPoint(alert.position);
        
        // Choose color based on severity (suspicious behavior disabled)
        QColor color = Qt::green;
        switch (alert.severity) {
            case casa_anzen::SecuritySeverity::LOW:
                color = Qt::yellow;
                break;
            case casa_anzen::SecuritySeverity::MEDIUM:
                continue; // Skip drawing medium severity alerts
            case casa_anzen::SecuritySeverity::HIGH:
                color = Qt::red;
                break;
            case casa_anzen::SecuritySeverity::CRITICAL:
            case casa_anzen::SecuritySeverity::EMERGENCY:
                color = QColor(128, 0, 128); // Purple
                break;
            default:
                color = Qt::cyan;
                break;
        }
        
        // Draw alert circle
        painter.setPen(QPen(color, 3));
        painter.setBrush(QBrush(color, Qt::SolidPattern));
        painter.drawEllipse(center, 20, 20);
        
        // Draw alert text
        QString alert_text = QString::fromStdString(alert.alert_type);
        painter.setPen(QPen(Qt::white, 1));
        painter.setFont(QFont("Arial", 10, QFont::Bold));
        QRect text_rect = painter.fontMetrics().boundingRect(alert_text);
        painter.drawText(center.x() - text_rect.width()/2, center.y() + text_rect.height()/2, alert_text);
    }
}

void VideoDisplayWidget::drawZones(QPainter& painter) {
    painter.setPen(QPen(Qt::white, 2, Qt::DashLine));
    
    for (const auto& zone : m_zones) {
        if (!zone.enabled) continue;
        
        QPolygon polygon;
        for (const auto& point : zone.polygon) {
            // Use the same mapping as other overlays to account for letterboxing offset
            polygon << cvPointToQPoint(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
        }
        
        painter.drawPolygon(polygon);
        
        // Draw zone name
        if (!polygon.isEmpty()) {
            QPoint center = polygon.boundingRect().center();
            painter.setPen(QPen(Qt::white, 1));
            painter.setFont(QFont("Arial", 12, QFont::Bold));
            painter.drawText(center, QString::fromStdString(zone.name));
        }
    }
}

void VideoDisplayWidget::drawSystemInfo(QPainter& painter) {
    // Draw system label and time at top-left
    painter.setPen(QPen(Qt::white, 1));
    painter.setFont(QFont("Arial", 14, QFont::Bold));
    QDateTime now = QDateTime::currentDateTime();
    QString timestamp = now.toString("yyyy-MM-dd hh:mm:ss");
    QString hud = QString("CASA ANZEN  %1").arg(timestamp);
    painter.drawText(10, 25, hud);
}

QRect VideoDisplayWidget::cvRectToQRect(const cv::Rect& rect) const {
    int x = static_cast<int>(rect.x * m_scale_x);
    int y = static_cast<int>(rect.y * m_scale_y);
    int width = static_cast<int>(rect.width * m_scale_x);
    int height = static_cast<int>(rect.height * m_scale_y);
    
    // Offset by the video position
    QSize widget_size = size();
    QSize frame_size(m_current_frame.cols, m_current_frame.rows);
    
    float scale_x = static_cast<float>(widget_size.width()) / frame_size.width();
    float scale_y = static_cast<float>(widget_size.height()) / frame_size.height();
    float scale = std::min(scale_x, scale_y);
    
    int scaled_width = static_cast<int>(frame_size.width() * scale);
    int scaled_height = static_cast<int>(frame_size.height() * scale);
    
    int offset_x = (widget_size.width() - scaled_width) / 2;
    int offset_y = (widget_size.height() - scaled_height) / 2;
    
    return QRect(x + offset_x, y + offset_y, width, height);
}

QPoint VideoDisplayWidget::cvPointToQPoint(const cv::Point2f& point) const {
    int x = static_cast<int>(point.x * m_scale_x);
    int y = static_cast<int>(point.y * m_scale_y);
    
    // Offset by the video position
    QSize widget_size = size();
    QSize frame_size(m_current_frame.cols, m_current_frame.rows);
    
    float scale_x = static_cast<float>(widget_size.width()) / frame_size.width();
    float scale_y = static_cast<float>(widget_size.height()) / frame_size.height();
    float scale = std::min(scale_x, scale_y);
    
    int scaled_width = static_cast<int>(frame_size.width() * scale);
    int scaled_height = static_cast<int>(frame_size.height() * scale);
    
    int offset_x = (widget_size.width() - scaled_width) / 2;
    int offset_y = (widget_size.height() - scaled_height) / 2;
    
    return QPoint(x + offset_x, y + offset_y);
}
cv::Point VideoDisplayWidget::mapDisplayPointToFrame(const QPoint& displayPoint) const {
    // Reverse mapping for a single point
    QSize widget_size = size();
    QSize frame_size(m_current_frame.cols, m_current_frame.rows);
    float scale_x = static_cast<float>(widget_size.width()) / frame_size.width();
    float scale_y = static_cast<float>(widget_size.height()) / frame_size.height();
    float scale = std::min(scale_x, scale_y);
    int scaled_width = static_cast<int>(frame_size.width() * scale);
    int scaled_height = static_cast<int>(frame_size.height() * scale);
    int offset_x = (widget_size.width() - scaled_width) / 2;
    int offset_y = (widget_size.height() - scaled_height) / 2;
    int x = std::clamp(displayPoint.x() - offset_x, 0, scaled_width);
    int y = std::clamp(displayPoint.y() - offset_y, 0, scaled_height);
    float inv = 1.0f / scale;
    return cv::Point(static_cast<int>(x * inv), static_cast<int>(y * inv));
}

QRect VideoDisplayWidget::normalizeRect(const QRect& rect) const {
    int x = rect.x();
    int y = rect.y();
    int w = rect.width();
    int h = rect.height();
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    return QRect(x, y, w, h);
}

cv::Rect VideoDisplayWidget::mapDisplayRectToFrame(const QRect& displayRect) const {
    // Reverse of cvRectToQRect: map from widget space (drawn image area) back to frame space
    QSize widget_size = size();
    QSize frame_size(m_current_frame.cols, m_current_frame.rows);
    float scale_x = static_cast<float>(widget_size.width()) / frame_size.width();
    float scale_y = static_cast<float>(widget_size.height()) / frame_size.height();
    float scale = std::min(scale_x, scale_y);
    int scaled_width = static_cast<int>(frame_size.width() * scale);
    int scaled_height = static_cast<int>(frame_size.height() * scale);
    int offset_x = (widget_size.width() - scaled_width) / 2;
    int offset_y = (widget_size.height() - scaled_height) / 2;
    int x = std::clamp(displayRect.x() - offset_x, 0, scaled_width);
    int y = std::clamp(displayRect.y() - offset_y, 0, scaled_height);
    int w = std::clamp(displayRect.width(), 0, scaled_width - x);
    int h = std::clamp(displayRect.height(), 0, scaled_height - y);
    // Scale back to frame coordinates
    float inv = 1.0f / scale;
    return cv::Rect(static_cast<int>(x * inv), static_cast<int>(y * inv),
                    static_cast<int>(w * inv), static_cast<int>(h * inv));
}

void VideoDisplayWidget::drawBoundingBox(QPainter& painter, const QRect& rect, const QColor& color, int thickness) {
    int corner_length = std::min(20, std::min(rect.width(), rect.height()) / 4);
    painter.setPen(QPen(color, thickness));
    
    // Top-left corner
    painter.drawLine(rect.left(), rect.top(), rect.left() + corner_length, rect.top());
    painter.drawLine(rect.left(), rect.top(), rect.left(), rect.top() + corner_length);
    
    // Top-right corner
    painter.drawLine(rect.right(), rect.top(), rect.right() - corner_length, rect.top());
    painter.drawLine(rect.right(), rect.top(), rect.right(), rect.top() + corner_length);
    
    // Bottom-left corner
    painter.drawLine(rect.left(), rect.bottom(), rect.left() + corner_length, rect.bottom());
    painter.drawLine(rect.left(), rect.bottom(), rect.left(), rect.bottom() - corner_length);
    
    // Bottom-right corner
    painter.drawLine(rect.right(), rect.bottom(), rect.right() - corner_length, rect.bottom());
    painter.drawLine(rect.right(), rect.bottom(), rect.right(), rect.bottom() - corner_length);
}

void VideoDisplayWidget::drawLabel(QPainter& painter, const QString& label, const QColor& color, int x, int y) {
    painter.setFont(QFont("Arial", 10, QFont::Bold));
    QRect text_rect = painter.fontMetrics().boundingRect(label);
    QPoint text_pos(x, y);
    QRect bg_rect = text_rect.translated(text_pos);
    bg_rect.adjust(-2, -2, 2, 2);
    
    // Draw semi-transparent background
    painter.fillRect(bg_rect, QColor(0, 0, 0, 180));
    painter.setPen(color);
    painter.drawText(text_pos, label);
}

QColor VideoDisplayWidget::getClassColor(const std::string& class_name) const {
    // Color mapping for different object classes
    if (class_name == "person") {
        return QColor(0, 255, 0);      // Green
    } else if (class_name == "car" || class_name == "truck" || class_name == "bus") {
        return QColor(0, 165, 255);    // Orange
    } else if (class_name == "bicycle" || class_name == "motorcycle") {
        return QColor(255, 255, 0);    // Yellow
    } else if (class_name == "dog" || class_name == "cat") {
        return QColor(255, 0, 255);    // Magenta
    } else {
        return QColor(0, 255, 255);    // Cyan (default)
    }
}

} // namespace casa_anzen
