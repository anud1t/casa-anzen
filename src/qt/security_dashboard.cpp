/**
 * @file security_dashboard.cpp
 * @brief Main security dashboard window implementation for Casa Anzen
 * @author Anudit Gautam
 */

#include "qt/security_dashboard.hpp"
#include "qt/video_display_widget.hpp"
#include "core/video_processing_thread.hpp"
#include "utils/logger.hpp"
#include <QApplication>
#include <QDir>
#include <QCoreApplication>
#include <QMenuBar>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QProgressBar>
#include <QTimer>
#include <QKeyEvent>
#include <QCloseEvent>
#include <QDockWidget>
#include <QListWidget>
#include <QSplitter>
#include <QPixmap>
#include <QScrollBar>
#include <QPushButton>
#include <QLineEdit>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QBuffer>
#include <QFile>
#include <QFrame>
#include <QSizePolicy>
#include <QStackedLayout>
#include <QFrame>
#include <QSizePolicy>

namespace casa_anzen {

SecurityDashboard::SecurityDashboard(QWidget *parent)
    : QMainWindow(parent)
    , m_confidence_threshold(0.25f)
    , m_recording_enabled(false)
    , m_debug_mode(false)
    , m_frame_count(0)
    , m_current_fps(0.0f)
    , m_update_timer(nullptr) {
    
    setupUI();
    setupConnections();
    setupMenuBar();
    
    // Initialize core components
    m_security_detector = std::make_unique<casa_anzen::SecurityDetector>();
    m_zone_manager = std::make_unique<casa_anzen::ZoneManager>();
    // Initialize SQLite and load any existing zones
    m_zone_manager->initializeDatabase("data/casa_anzen.db");
    m_recording_manager = std::make_unique<casa_anzen::RecordingManager>();
    
    // Initialize logger
    casa_anzen::Logger::getInstance().setLogLevel(casa_anzen::Logger::Level::INFO);
    casa_anzen::Logger::getInstance().enableConsoleOutput(true);
    
    // Network
    m_network = new QNetworkAccessManager(this);
    
    // Set window properties
    setWindowTitle("Casa Anzen Security System");
    setMinimumSize(1200, 800);
    resize(1600, 1000);
}
static QString captureDirPath() {
    // Resolve to project root's data/captures: executable is in build/, so go up one
    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp();
    QString path = appDir.filePath("data/captures");
    return path;
}

static QString captureSubdirPath(const QString& sub) {
    QDir root(captureDirPath());
    return root.filePath(sub);
}

void SecurityDashboard::onNewFrame(const cv::Mat& frame) {
    m_last_frame = frame.clone();
}

SecurityDashboard::~SecurityDashboard() {
    stopProcessing();
}

void SecurityDashboard::setupUI() {
    // Create central widget
    m_central_widget = new QWidget(this);
    setCentralWidget(m_central_widget);
    
    // Create main layout
    m_main_layout = new QVBoxLayout(m_central_widget);
    
    // Create video display widget
    m_video_display = new VideoDisplayWidget(this);
    m_main_layout->addWidget(m_video_display);

    // Create collapsible side panel as dock (no title)
    m_side_dock = new QDockWidget("", this);
    m_side_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    m_side_dock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    // Hide the dock title bar for a cleaner look
    m_side_dock->setTitleBarWidget(new QWidget(m_side_dock));
    m_side_panel = new QWidget(m_side_dock);
    m_side_layout = new QVBoxLayout(m_side_panel);
    m_events_header = new QLabel("Event Feed");
    m_events_header->setStyleSheet("font-weight:600; font-size:14px; color:#cccccc; padding:6px 4px;");
    m_side_layout->addWidget(m_events_header);

    // Event toolbar (VIEW, CAPTION, DELETE, DELETE ALL)
    m_event_toolbar = new QWidget(m_side_panel);
    {
        QHBoxLayout* tl = new QHBoxLayout(m_event_toolbar);
        tl->setContentsMargins(8,4,8,4);
        tl->setSpacing(8);
        m_view_btn = new QPushButton("VIEW", m_event_toolbar);
        m_caption_btn = new QPushButton("CAPTION", m_event_toolbar);
        m_delete_btn = new QPushButton("DELETE", m_event_toolbar);
        m_delete_all_btn = new QPushButton("DELETE ALL", m_event_toolbar);
        m_view_btn->setCursor(Qt::PointingHandCursor);
        m_caption_btn->setCursor(Qt::PointingHandCursor);
        m_delete_btn->setCursor(Qt::PointingHandCursor);
        m_delete_all_btn->setCursor(Qt::PointingHandCursor);
        m_event_toolbar->setStyleSheet(
            "QWidget{ background:#262626; border:1px solid #3a3a3a; border-radius:6px;}"
            "QPushButton{ background-color:#3c3c3c; color:#ffffff; border:1px solid #555; padding:6px 10px; border-radius:4px; min-height:28px;}"
            "QPushButton:hover{ background-color:#4a4a4a;}"
            "QPushButton#Danger{ background-color:#7a1f1f; border:1px solid #9a2a2a;}"
            "QPushButton#Danger:hover{ background-color:#8a2727;}"
        );
        m_event_toolbar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        m_event_toolbar->setMinimumHeight(44);
        m_delete_all_btn->setObjectName("Danger");
        tl->addWidget(m_view_btn);
        tl->addWidget(m_caption_btn);
        tl->addWidget(m_delete_btn);
        tl->addStretch();
        tl->addWidget(m_delete_all_btn);
        m_event_toolbar->setLayout(tl);
    }
    m_side_layout->addWidget(m_event_toolbar);
    m_event_list = new QListWidget(m_side_panel);
    m_event_list->setViewMode(QListView::ListMode);
    m_event_list->setResizeMode(QListView::Adjust);
    m_event_list->setUniformItemSizes(false);
    m_event_list->setSpacing(14);
    m_event_list->setMovement(QListView::Static);
    m_event_list->setSelectionMode(QAbstractItemView::SingleSelection);
    m_event_list->setStyleSheet("QListWidget{ background:#1f1f1f; border:1px solid #3a3a3a;} QListWidget::item{ color:#dddddd; border:none; }");
    m_side_layout->addWidget(m_event_list, 1);

    // Zones controls
    m_zones_header = new QLabel("Zones");
    m_zones_header->setStyleSheet("font-weight:600; font-size:14px; color:#cccccc; padding:6px 4px;");
    m_side_layout->addWidget(m_zones_header);
    m_zone_controls_box = new QWidget(m_side_panel);
    auto zcLayout = new QHBoxLayout(m_zone_controls_box);
    zcLayout->setContentsMargins(0,0,0,0);
    m_zone_name_edit = new QLineEdit(m_zone_controls_box);
    m_zone_name_edit->setPlaceholderText("Zone label (optional)");
    m_draw_btn = new QPushButton("Draw Zone", m_zone_controls_box);
    m_draw_btn->setToolTip("Click to enable drawing mode, then draw on video");
    m_clear_zones_btn = new QPushButton("Clear Zones", m_zone_controls_box);
    m_toggle_zones_btn = new QPushButton("Hide Zones", m_zone_controls_box);
    zcLayout->addWidget(m_zone_name_edit, 1);
    zcLayout->addWidget(m_draw_btn);
    zcLayout->addWidget(m_clear_zones_btn);
    zcLayout->addWidget(m_toggle_zones_btn);
    m_zone_controls_box->setLayout(zcLayout);
    m_zone_controls_box->setStyleSheet("QWidget{background:#2b2b2b; border:1px solid #3a3a3a; border-radius:6px; padding:6px;}");
    m_side_layout->addWidget(m_zone_controls_box);
    m_side_panel->setLayout(m_side_layout);
    m_side_dock->setWidget(m_side_panel);
    addDockWidget(Qt::RightDockWidgetArea, m_side_dock);
    
    // Create status bar
    createStatusBar();
}

void SecurityDashboard::setupConnections() {
    // Update timer for status updates
    m_update_timer = new QTimer(this);
    connect(m_update_timer, &QTimer::timeout, this, &SecurityDashboard::updateStatus);
    m_update_timer->start(1000); // Update every second
    // Side panel: draw zone button toggles draw mode
    connect(m_draw_btn, &QPushButton::clicked, this, [this](){
        if (!m_video_display) return;
        bool enable = !m_video_display->isDrawModeEnabled();
        m_video_display->setDrawModeEnabled(enable);
        m_draw_btn->setText(enable ? "Finish Drawing" : "Draw Zone");
        m_status_label->setText(enable ? "DRAW MODE: ON" : "DRAW MODE: OFF");
    });

    // Side panel: clear zones
    connect(m_clear_zones_btn, &QPushButton::clicked, this, [this](){
        if (!m_zone_manager) return;
        m_zone_manager->clearZones();
        m_zone_manager->saveZonesToDatabase();
        if (m_video_display) m_video_display->setZoneOverlays(m_zone_manager->getAllZones());
        m_status_label->setText("All zones cleared");
    });

    // Side panel: hide/show zones overlays
    connect(m_toggle_zones_btn, &QPushButton::clicked, this, [this](){
        m_zones_visible = !m_zones_visible;
        if (m_video_display) m_video_display->setZoneOverlays(m_zones_visible ? m_zone_manager->getAllZones() : std::vector<casa_anzen::SecurityZone>{});
        m_toggle_zones_btn->setText(m_zones_visible ? "Hide Zones" : "Show Zones");
    });

    // Event list: preview full-size image on double click
    connect(m_event_list, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item){
        previewItem(item);
    });

    // Toolbar actions
    connect(m_view_btn, &QPushButton::clicked, this, [this]() {
        previewItem(m_event_list->currentItem());
    });
    connect(m_caption_btn, &QPushButton::clicked, this, [this]() {
        captionItem(m_event_list->currentItem());
    });
    connect(m_delete_btn, &QPushButton::clicked, this, [this]() {
        deleteItem(m_event_list->currentItem());
    });
    connect(m_delete_all_btn, &QPushButton::clicked, this, [this]() {
        deleteAllCaptures();
    });

    // Event list: selection changed -> update visual highlight
    connect(m_event_list, &QListWidget::itemSelectionChanged, this, [this]() {
        for (int i = 0; i < m_event_list->count(); ++i) {
            QListWidgetItem* it = m_event_list->item(i);
            QWidget* w = m_event_list->itemWidget(it);
            if (!w) continue;
            bool selected = (it == m_event_list->currentItem());
            // Card is a QFrame; toggle yellow outline when selected
            if (QFrame* frame = qobject_cast<QFrame*>(w)) {
                frame->setStyleSheet(QString(
                    "#eventCard{ background:#232323; border:2px solid %1; border-radius:8px;}"
                ).arg(selected ? "#e0c341" : "#3a3a3a"));
            }
        }
    });

    // Capture zone created events from the video display
    connect(m_video_display, &VideoDisplayWidget::zoneCreated, this, [this](const casa_anzen::SecurityZone& zoneIn, const cv::Mat& frame){
        // Add to ZoneManager
        casa_anzen::SecurityZone zone = zoneIn;
        // Use user-provided label if provided, else sequential
        QString labelText = m_zone_name_edit ? m_zone_name_edit->text().trimmed() : QString();
        if (!labelText.isEmpty()) {
            zone.name = labelText.toStdString();
        } else {
            ++m_zone_seq;
            zone.name = std::string("Zone ") + std::to_string(m_zone_seq);
        }
        m_zone_manager->addZone(zone);
        // Persist zones to SQLite
        m_zone_manager->saveZonesToDatabase();
        // Update overlays
        m_video_display->setZoneOverlays(m_zone_manager->getAllZones());
        if (m_zone_name_edit) m_zone_name_edit->clear();
        if (m_video_display && m_video_display->isDrawModeEnabled()) {
            m_video_display->setDrawModeEnabled(false);
            if (m_draw_btn) m_draw_btn->setText("Draw Zone");
        }
        // Save capture
        try { QDir().mkpath(captureSubdirPath("zones")); } catch (...) {}
        // Crop and save (use bounding rect; support lines too with 2 points)
        if (zone.polygon.size() >= 2) {
            cv::Rect r = cv::boundingRect(zone.polygon);
            cv::Rect bounded(0, 0, frame.cols, frame.rows);
            r = r & bounded;
            if (r.width > 0 && r.height > 0) {
                cv::Mat crop = frame(r).clone();
                auto ts = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmsszzz");
                std::string path = captureSubdirPath("zones").toStdString() + "/" + zone.name + "_created_" + ts.toStdString() + ".jpg";
                cv::imwrite(path, crop);
            }
        }
        m_status_label->setText("Zone created: " + QString::fromStdString(zone.name));
    });

    // Click-to-capture from the video widget
    connect(m_video_display, &VideoDisplayWidget::captureRequested, this,
            [this](const QString& class_name, const cv::Rect& bbox, const cv::Mat& frame){
        if (frame.empty() || bbox.width <= 0 || bbox.height <= 0) return;
        cv::Rect bounded(0, 0, frame.cols, frame.rows);
        cv::Rect r = bbox & bounded;
        if (r.width <= 0 || r.height <= 0) return;
        cv::Mat crop = frame(r).clone();
        auto ts = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmsszzz");
        bool isPerson = (class_name.toLower() == "person");
        bool isVehicle = (class_name.toLower() == "car" || class_name.toLower() == "truck" ||
                          class_name.toLower() == "bus" || class_name.toLower() == "motorcycle" ||
                          class_name.toLower() == "bicycle");
        const char* sub = isPerson ? "persons" : (isVehicle ? "vehicles" : "misc");
        try { QDir().mkpath(captureSubdirPath(sub)); } catch (...) {}
        std::string fname = std::string("CLICK_") + class_name.toUpper().toStdString() + "_" + ts.toStdString() + ".jpg";
        std::string path = captureSubdirPath(sub).toStdString() + "/" + fname;
        cv::imwrite(path, crop);
        m_status_label->setText("Saved click capture: " + QString::fromStdString(fname));

        // Add thumbnail to event feed (correct RGB)
        cv::Mat rgb;
        cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
        QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        QPixmap pix = QPixmap::fromImage(img).scaled(200, 112, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        QWidget* card = createEventCard(QString("CLICK: ") + class_name.toUpper(), pix);
        QListWidgetItem* item = new QListWidgetItem();
        // Ensure enough space for caption wrapping regardless of thumbnail shape
        QSize hint = card->sizeHint();
        hint.setHeight(std::max(hint.height(), 190));
        item->setSizeHint(hint);
        item->setData(Qt::UserRole, QString::fromStdString(path));
        m_event_list->insertItem(0, item);
        m_event_list->setItemWidget(item, card);
    });
}
void SecurityDashboard::previewItem(QListWidgetItem* item) {
    if (!item) return;
    QString path = item->data(Qt::UserRole).toString();
    if (path.isEmpty()) return;
    QImage img(path);
    if (img.isNull()) return;
    QDialog* dlg = new QDialog(this);
    dlg->setWindowTitle("Preview");
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    QVBoxLayout* lay = new QVBoxLayout(dlg);
    QLabel* lbl = new QLabel(dlg);
    lbl->setAlignment(Qt::AlignCenter);
    QSize maxSize(1000, 700);
    QPixmap px = QPixmap::fromImage(img);
    if (px.width() > maxSize.width() || px.height() > maxSize.height()) {
        px = px.scaled(maxSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    lbl->setPixmap(px);
    lay->addWidget(lbl);
    dlg->resize(px.size());
    dlg->show();
}

void SecurityDashboard::captionItem(QListWidgetItem* item) {
    if (!item) return;
    QString path = item->data(Qt::UserRole).toString();
    if (path.isEmpty()) return;

    QImage image(path);
    if (image.isNull()) return;
    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "JPEG");
    QByteArray base64 = bytes.toBase64();
    QString dataUrl = QStringLiteral("data:image/jpeg;base64,") + QString::fromLatin1(base64);

    QJsonObject payload;
    payload.insert("image_url", dataUrl);
    payload.insert("length", "short");
    QJsonDocument doc(payload);
    QByteArray body = doc.toJson(QJsonDocument::Compact);

    QNetworkRequest req(QUrl("http://localhost:2020/v1/caption"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    if (QWidget* w = m_event_list->itemWidget(item)) {
        if (QLabel* cap = w->findChild<QLabel*>("captionLabel")) {
            cap->setVisible(true);
            cap->setText(QString::fromUtf8("… captioning …"));
        }
    }

    QNetworkReply* reply = m_network->post(req, body);
    connect(reply, &QNetworkReply::finished, this, [this, reply, item]() {
        reply->deleteLater();
        QLabel* cap = nullptr;
        if (QWidget* w = m_event_list->itemWidget(item)) cap = w->findChild<QLabel*>("captionLabel");
        if (reply->error() != QNetworkReply::NoError) {
            if (cap) { cap->setVisible(true); cap->setText("(caption failed)"); }
            return;
        }
        QByteArray resp = reply->readAll();
        QJsonParseError err;
        QJsonDocument jdoc = QJsonDocument::fromJson(resp, &err);
        if (err.error != QJsonParseError::NoError || !jdoc.isObject()) {
            if (cap) { cap->setVisible(true); cap->setText("(caption parse error)"); }
            return;
        }
        QString caption;
        QJsonObject obj = jdoc.object();
        if (obj.contains("caption")) {
            caption = obj.value("caption").toString();
        } else if (obj.contains("data") && obj.value("data").isObject()) {
            caption = obj.value("data").toObject().value("caption").toString();
        }
        if (caption.isEmpty()) caption = "(no caption)";
        if (cap) { cap->setVisible(true); cap->setText(caption); }
    });
}

void SecurityDashboard::deleteItem(QListWidgetItem* item) {
    if (!item) return;
    QString path = item->data(Qt::UserRole).toString();
    if (!path.isEmpty()) {
        QFile f(path);
        if (f.exists()) {
            f.remove();
        }
    }
    int row = m_event_list->row(item);
    QListWidgetItem* removed = m_event_list->takeItem(row);
    delete removed;
}

void SecurityDashboard::deleteAllCaptures() {
    // Remove files under data/captures recursively
    QDir capRoot(captureDirPath());
    if (capRoot.exists()) {
        capRoot.removeRecursively();
        // recreate base dir to avoid issues elsewhere
        QDir().mkpath(captureDirPath());
    }
    // Clear UI list
    m_event_list->clear();
    m_status_label->setText("All captures deleted");
}


QWidget* SecurityDashboard::createEventCard(const QString& title, const QPixmap& thumbnail, const QString& caption) {
    QFrame* card = new QFrame(m_event_list);
card->setObjectName("eventCard");
card->setFrameShape(QFrame::NoFrame);
card->setStyleSheet("#eventCard{ background:#232323; border:1px solid #3a3a3a; border-radius:8px; }");
    QVBoxLayout* v = new QVBoxLayout(card);
    v->setContentsMargins(10,10,10,10);
    v->setSpacing(6);

    // Title badge ABOVE the thumbnail (not overlapping)
    QLabel* badge = new QLabel(title.toUpper(), card);
    badge->setAlignment(Qt::AlignHCenter);
    QFont badgeFont = badge->font();
    badgeFont.setBold(true);
    badgeFont.setPointSize(std::max(10, badgeFont.pointSize()));
    badge->setFont(badgeFont);
    badge->setMinimumHeight(22);
    badge->setStyleSheet("color:#ffffff; background:#2b2b2b; padding:4px 12px; border-radius:12px;");
    v->addWidget(badge, 0, Qt::AlignHCenter);

    // Thumbnail under the badge
    QLabel* thumb = new QLabel(card);
    thumb->setObjectName("thumbLabel");
    thumb->setAlignment(Qt::AlignCenter);
    thumb->setPixmap(thumbnail);
    thumb->setStyleSheet("border:none;");
    thumb->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    thumb->setMinimumHeight(120);
    v->addWidget(thumb);

    QLabel* capLbl = new QLabel(caption, card);
    capLbl->setObjectName("captionLabel");
    capLbl->setWordWrap(true);
    capLbl->setStyleSheet("color:#bfbfbf; background:#1d1d1d; border:1px solid #343434; border-radius:6px; padding:8px; font-size:12px;");
    capLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
    capLbl->setVisible(!caption.isEmpty());
    v->addWidget(capLbl);
    v->addStretch(1); // Push following items down if any

    // Initial geometry
    capLbl->adjustSize();

    return card;
}

void SecurityDashboard::setupMenuBar() {
    // Intentionally empty: remove all menus for a clean viewer UI
}

void SecurityDashboard::createStatusBar() {
    m_status_bar = statusBar();
    
    // Status label
    m_status_label = new QLabel("Ready");
    m_status_bar->addWidget(m_status_label);
    
    // Mode label (detection profile)
    m_mode_label = new QLabel("MODE: PEOPLE + VEHICLES");
    m_status_bar->addPermanentWidget(m_mode_label);
    
    // FPS label
    m_fps_label = new QLabel("FPS: 0");
    m_status_bar->addPermanentWidget(m_fps_label);
    
    // Detections label
    m_detections_label = new QLabel("Detections: 0");
    m_status_bar->addPermanentWidget(m_detections_label);
    
    // Alerts label
    m_alerts_label = new QLabel("Alerts: 0");
    m_status_bar->addPermanentWidget(m_alerts_label);
    
    // Recording label
    m_recording_label = new QLabel("Recording: OFF");
    m_status_bar->addPermanentWidget(m_recording_label);
}

void SecurityDashboard::setModelPath(const std::string& model_path) {
    m_model_path = model_path;
    casa_anzen::Logger::getInstance().info("Model path set to: " + model_path);
}

void SecurityDashboard::setVideoSource(const std::string& video_source) {
    m_video_source = video_source;
    casa_anzen::Logger::getInstance().info("Video source set to: " + video_source);
}

void SecurityDashboard::setConfidenceThreshold(float threshold) {
    m_confidence_threshold = threshold;
    casa_anzen::Logger::getInstance().info("Confidence threshold set to: " + std::to_string(threshold));
}

void SecurityDashboard::enableRecording(bool enable) {
    m_recording_enabled = enable;
    m_recording_label->setText(QString("Recording: %1").arg(enable ? "ON" : "OFF"));
    casa_anzen::Logger::getInstance().info("Recording " + std::string(enable ? "enabled" : "disabled"));
}

void SecurityDashboard::enableDebugMode(bool enable) {
    m_debug_mode = enable;
    casa_anzen::Logger::getInstance().setLogLevel(enable ? 
        casa_anzen::Logger::Level::DEBUG : casa_anzen::Logger::Level::INFO);
    casa_anzen::Logger::getInstance().info("Debug mode " + std::string(enable ? "enabled" : "disabled"));
}

void SecurityDashboard::setRtspLatency(int latency_ms) {
    m_rtsp_latency = latency_ms;
    casa_anzen::Logger::getInstance().info("RTSP latency set to " + std::to_string(latency_ms) + "ms");
}

void SecurityDashboard::setRtspBufferSize(int buffer_size) {
    m_rtsp_buffer_size = buffer_size;
    casa_anzen::Logger::getInstance().info("RTSP buffer size set to " + std::to_string(buffer_size));
}

void SecurityDashboard::setRtspQuality(const std::string& quality) {
    m_rtsp_quality = quality;
    casa_anzen::Logger::getInstance().info("RTSP quality set to " + quality);
}

void SecurityDashboard::autoStartIfConfigured() {
    if (!m_model_path.empty() && !m_video_source.empty()) {
        casa_anzen::Logger::getInstance().info("Auto-starting processing with configured parameters");
        startProcessing();
    }
}

void SecurityDashboard::startProcessing() {
    if (m_processing_thread && m_processing_thread->isRunning()) {
        casa_anzen::Logger::getInstance().warning("Processing already running");
        return;
    }
    
    if (m_model_path.empty() || m_video_source.empty()) {
        QMessageBox::warning(this, "Configuration Error", 
                           "Please configure model path and video source before starting.");
        return;
    }
    
    // Create processing thread
    m_processing_thread = std::make_unique<VideoProcessingThread>(this);
    
    // Configure thread
    m_processing_thread->setModelPath(m_model_path);
    m_processing_thread->setVideoSource(m_video_source);
    m_processing_thread->setConfidenceThreshold(m_confidence_threshold);
    m_processing_thread->setSecurityDetector(m_security_detector.get());
    m_processing_thread->setZoneManager(m_zone_manager.get());
    m_processing_thread->setRecordingManager(m_recording_manager.get());
    m_processing_thread->setRtspLatency(m_rtsp_latency);
    m_processing_thread->setRtspBufferSize(m_rtsp_buffer_size);
    m_processing_thread->setRtspQuality(m_rtsp_quality);
    // Default to PEOPLE + VEHICLES
    m_current_mode = 2;
    m_processing_thread->setDetectionMode(m_current_mode);
    
    // Connect signals
    connect(m_processing_thread.get(), &VideoProcessingThread::newFrame,
            this, &SecurityDashboard::onNewFrame);
    connect(m_processing_thread.get(), &VideoProcessingThread::newFrame,
            m_video_display, &VideoDisplayWidget::updateFrame);
    connect(m_processing_thread.get(), &VideoProcessingThread::detectionData,
            this, &SecurityDashboard::updateDetectionData);
    connect(m_processing_thread.get(), &VideoProcessingThread::securityAlerts,
            this, &SecurityDashboard::updateSecurityAlerts);
    connect(m_processing_thread.get(), &VideoProcessingThread::processingError,
            this, &SecurityDashboard::handleProcessingError);
    connect(m_processing_thread.get(), &VideoProcessingThread::processingFinished,
            this, &SecurityDashboard::stopProcessing);
    
    // Start processing
    m_processing_thread->start();
    m_status_label->setText("Processing...");
    
    casa_anzen::Logger::getInstance().info("Started video processing");
}

void SecurityDashboard::stopProcessing() {
    if (m_processing_thread && m_processing_thread->isRunning()) {
        m_processing_thread->stop();
        m_processing_thread->wait();
        m_processing_thread.reset();
        
        m_status_label->setText("Stopped");
        casa_anzen::Logger::getInstance().info("Stopped video processing");
    }
}

void SecurityDashboard::openVideoFile() {
    QString filename = QFileDialog::getOpenFileName(this, "Open Video File", "",
        "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)");
    
    if (!filename.isEmpty()) {
        setVideoSource(filename.toStdString());
    }
}

void SecurityDashboard::openModelFile() {
    QString filename = QFileDialog::getOpenFileName(this, "Open Model File", "",
        "Model Files (*.engine *.onnx);;All Files (*)");
    
    if (!filename.isEmpty()) {
        setModelPath(filename.toStdString());
    }
}

void SecurityDashboard::handleProcessingError(const QString& error_message) {
    QMessageBox::critical(this, "Processing Error", error_message);
    stopProcessing();
}

void SecurityDashboard::updateStatus() {
    if (m_processing_thread && m_processing_thread->isRunning()) {
        m_current_fps = m_processing_thread->getCurrentFPS();
        m_frame_count = m_processing_thread->getFrameCount();
        
        m_fps_label->setText(QString("FPS: %1").arg(m_current_fps, 0, 'f', 1));
        m_detections_label->setText(QString("Detections: %1").arg(m_current_detections.size()));
        m_alerts_label->setText(QString("Alerts: %1").arg(m_current_alerts.size()));
    }
}

void SecurityDashboard::updateDetectionData(const std::vector<casa_anzen::TrackedObject>& tracks,
                                          const std::vector<casa_anzen::Detection>& detections) {
    m_current_tracks = tracks;
    m_current_detections = detections;
    
    // Update video display overlays
    m_video_display->setDetectionOverlays(tracks, detections);

    // Event feed now only updates on zone entry (below) or manual clicks

    // Edge-triggered zone entry capture for people/vehicles
    if (!m_current_tracks.empty()) {
        auto zones = m_zone_manager->getAllZones();
        if (!zones.empty()) {
            // Ensure subdirectories exist
            try { QDir().mkpath(captureSubdirPath("persons")); } catch (...) {}
            try { QDir().mkpath(captureSubdirPath("vehicles")); } catch (...) {}
            for (const auto& zone : zones) {
                if (!zone.enabled) continue;
                // Build zone rect from polygon or line (supports >=2 points)
                if (zone.polygon.size() >= 2) {
                    cv::Rect zrect = cv::boundingRect(zone.polygon);
                    // Hysteresis thresholds for overlap w.r.t. track bbox
                    const float overlapEnter = 0.15f; // 15% of bbox inside zone to count as entry
                    const float overlapExit  = 0.05f; // drop-out threshold

                    for (const auto& t : m_current_tracks) {
                        // only people/vehicles
                        const std::string& cls = t.class_name;
                        bool isPerson = (cls == "person");
                        bool isVehicle = (cls == "car" || cls == "truck" || cls == "bus" || cls == "motorcycle" || cls == "bicycle");
                        if (!isPerson && !isVehicle) continue;
                        cv::Rect tb(static_cast<int>(t.bbox.x), static_cast<int>(t.bbox.y),
                                    static_cast<int>(t.bbox.width), static_cast<int>(t.bbox.height));
                        // Overlap ratio between track bbox and zone rect, normalized by track bbox area
                        cv::Rect inter = tb & zrect;
                        float overlapRatio = 0.0f;
                        if (tb.width > 0 && tb.height > 0 && inter.width > 0 && inter.height > 0) {
                            overlapRatio = static_cast<float>(inter.width * inter.height) /
                                           static_cast<float>(tb.width * tb.height);
                        }

                        // Hysteresis: require higher overlap to enter than to remain
                        bool inside = overlapRatio >= overlapEnter;
                        std::string key = zone.name + ":" + std::to_string(t.track_id);
                        bool wasInside = m_inside_keys.count(key) > 0;
                        // If overlap is between exit and enter and was inside, keep it inside
                        if (!inside && wasInside && overlapRatio > overlapExit) {
                            inside = true;
                        }

                        if (inside && !wasInside && !m_last_frame.empty()) {
                            // Entered: capture crop of track bbox
                            cv::Rect bounded(0, 0, m_last_frame.cols, m_last_frame.rows);
                            cv::Rect r = tb & bounded;
                            if (r.width > 0 && r.height > 0) {
                                cv::Mat crop = m_last_frame(r).clone();
                                auto ts = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmsszzz");
                                std::string base = isPerson ? "PERSON" : "VEHICLE";
                                const char* sub = isPerson ? "persons" : "vehicles";
                                std::string path = captureSubdirPath(sub).toStdString() + "/" + zone.name + "_" + base + "_id" + std::to_string(t.track_id) + "_" + ts.toStdString() + ".jpg";
                                cv::imwrite(path, crop);

                                // Add thumbnail to event feed (Zone entry)
                                cv::Mat rgb;
                                cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
                                QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
                                QPixmap pix = QPixmap::fromImage(img).scaled(200, 112, Qt::KeepAspectRatio, Qt::SmoothTransformation);
                                QString label = QString::fromStdString(zone.name).toUpper() + ": " + (isPerson ? "PERSON" : "VEHICLE");
                                QWidget* card = createEventCard(label, pix);
                                QListWidgetItem* item = new QListWidgetItem();
                                QSize hint = card->sizeHint();
                                hint.setHeight(std::max(hint.height(), 190));
                                item->setSizeHint(hint);
                                item->setData(Qt::UserRole, QString::fromStdString(path));
                                m_event_list->insertItem(0, item);
                                m_event_list->setItemWidget(item, card);
                            }
                            m_inside_keys.insert(key);
                        } else if (!inside && wasInside) {
                            // Exited
                            m_inside_keys.erase(key);
                        }
                    }
                }
            }
        }
    }
}

void SecurityDashboard::updateSecurityAlerts(const std::vector<casa_anzen::SecurityAlert>& alerts) {
    m_current_alerts = alerts;
    
    // Update video display overlays
    m_video_display->setAlertOverlays(alerts);
    
    // Log critical alerts
    for (const auto& alert : alerts) {
        if (alert.severity >= casa_anzen::SecuritySeverity::HIGH) {
            casa_anzen::Logger::getInstance().warning("High priority alert: " + alert.message);
        }
    }
}

void SecurityDashboard::toggleFullscreen() {
    if (isFullScreen()) {
        showNormal();
    } else {
        showFullScreen();
    }
}

// Removed unused dialogs

void SecurityDashboard::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Space:
            if (m_processing_thread && m_processing_thread->isRunning()) {
                stopProcessing();
            } else {
                startProcessing();
            }
            break;
        case Qt::Key_F11:
            toggleFullscreen();
            break;
        case Qt::Key_Escape:
            if (isFullScreen()) {
                showNormal();
            }
            break;
        case Qt::Key_1:
            // PEOPLE
            if (m_processing_thread && m_processing_thread->isRunning()) {
                m_current_mode = 0;
                m_processing_thread->setDetectionMode(m_current_mode);
            }
            m_status_label->setText("MODE: PEOPLE");
            if (m_mode_label) m_mode_label->setText("MODE: PEOPLE");
            break;
        case Qt::Key_2:
            // VEHICLES
            if (m_processing_thread && m_processing_thread->isRunning()) {
                m_current_mode = 1;
                m_processing_thread->setDetectionMode(m_current_mode);
            }
            m_status_label->setText("MODE: VEHICLES");
            if (m_mode_label) m_mode_label->setText("MODE: VEHICLES");
            break;
        case Qt::Key_3:
            // PEOPLE + VEHICLES
            if (m_processing_thread && m_processing_thread->isRunning()) {
                m_current_mode = 2;
                m_processing_thread->setDetectionMode(m_current_mode);
            }
            m_status_label->setText("MODE: PEOPLE + VEHICLES");
            if (m_mode_label) m_mode_label->setText("MODE: PEOPLE + VEHICLES");
            break;
        case Qt::Key_4:
            // ALL
            if (m_processing_thread && m_processing_thread->isRunning()) {
                m_current_mode = 3;
                m_processing_thread->setDetectionMode(m_current_mode);
            }
            m_status_label->setText("MODE: ALL");
            if (m_mode_label) m_mode_label->setText("MODE: ALL");
            break;
        case Qt::Key_Z:
            // Toggle draw mode
            if (m_video_display) {
                bool enabled = !m_video_display->isDrawModeEnabled();
                m_video_display->setDrawModeEnabled(enabled);
                m_status_label->setText(enabled ? "DRAW MODE: ON" : "DRAW MODE: OFF");
            }
            break;
        default:
            QMainWindow::keyPressEvent(event);
            break;
    }
}

void SecurityDashboard::closeEvent(QCloseEvent* event) {
    stopProcessing();
    event->accept();
}

} // namespace casa_anzen
