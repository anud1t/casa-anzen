#include "image_viewer_dialog.hpp"
#include <QFileInfo>
#include <QMessageBox>
#include <QApplication>
#include <QScreen>
#include <QDebug>

ImageViewerDialog::ImageViewerDialog(const QString& imagePath, QWidget* parent)
    : QDialog(parent)
    , m_mainLayout(nullptr)
    , m_buttonLayout(nullptr)
    , m_scrollArea(nullptr)
    , m_imageLabel(nullptr)
    , m_closeButton(nullptr)
    , m_zoomInButton(nullptr)
    , m_zoomOutButton(nullptr)
    , m_fitButton(nullptr)
    , m_scaleFactor(1.0)
{
    try {
        setupUI();
        applyMilitaryTheme();
        loadImage(imagePath);
    } catch (const std::exception& e) {
        qDebug() << "ImageViewerDialog constructor exception:" << e.what();
        throw;
    } catch (...) {
        qDebug() << "ImageViewerDialog constructor unknown exception";
        throw;
    }
}

void ImageViewerDialog::setupUI()
{
    setWindowTitle("Image Viewer");
    setModal(true);
    resize(800, 600);
    
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(8, 8, 8, 8);
    m_mainLayout->setSpacing(8);
    
    // Button layout
    m_buttonLayout = new QHBoxLayout();
    m_buttonLayout->setSpacing(8);
    
    m_zoomInButton = new QPushButton("Zoom In", this);
    m_zoomOutButton = new QPushButton("Zoom Out", this);
    m_fitButton = new QPushButton("Fit to Window", this);
    m_closeButton = new QPushButton("Close", this);
    
    m_buttonLayout->addWidget(m_zoomInButton);
    m_buttonLayout->addWidget(m_zoomOutButton);
    m_buttonLayout->addWidget(m_fitButton);
    m_buttonLayout->addStretch();
    m_buttonLayout->addWidget(m_closeButton);
    
    m_mainLayout->addLayout(m_buttonLayout);
    
    // Scroll area for image
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(true);
    m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    
    m_imageLabel = new QLabel(this);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setScaledContents(false);
    m_imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    
    m_scrollArea->setWidget(m_imageLabel);
    m_mainLayout->addWidget(m_scrollArea);
    
    // Connect signals
    connect(m_zoomInButton, &QPushButton::clicked, this, &ImageViewerDialog::onZoomIn);
    connect(m_zoomOutButton, &QPushButton::clicked, this, &ImageViewerDialog::onZoomOut);
    connect(m_fitButton, &QPushButton::clicked, this, &ImageViewerDialog::onFitToWindow);
    connect(m_closeButton, &QPushButton::clicked, this, &ImageViewerDialog::onClose);
}

void ImageViewerDialog::applyMilitaryTheme()
{
    setStyleSheet(
        "QDialog { "
        "background: #1a1a1a; "
        "color: #cccccc; "
        "border: 2px solid #00ff00; "
        "}"
    );
    
    m_zoomInButton->setStyleSheet(
        "QPushButton { "
        "background: #0d4d0d; "
        "color: #00ff00; "
        "border: 1px solid #00aa00; "
        "padding: 6px 12px; "
        "font-weight: bold; "
        "font-family: 'Courier New', monospace; "
        "}"
        "QPushButton:hover { "
        "background: #0f5f0f; "
        "}"
        "QPushButton:pressed { "
        "background: #0a3a0a; "
        "}"
    );
    
    m_zoomOutButton->setStyleSheet(m_zoomInButton->styleSheet());
    m_fitButton->setStyleSheet(m_zoomInButton->styleSheet());
    
    m_closeButton->setStyleSheet(
        "QPushButton { "
        "background: #4d0d0d; "
        "color: #ff0000; "
        "border: 1px solid #aa0000; "
        "padding: 6px 12px; "
        "font-weight: bold; "
        "font-family: 'Courier New', monospace; "
        "}"
        "QPushButton:hover { "
        "background: #5f0f0f; "
        "}"
        "QPushButton:pressed { "
        "background: #3a0a0a; "
        "}"
    );
    
    m_scrollArea->setStyleSheet(
        "QScrollArea { "
        "background: #0a0a0a; "
        "border: 1px solid #333333; "
        "}"
    );
}

void ImageViewerDialog::loadImage(const QString& imagePath)
{
    QFileInfo fileInfo(imagePath);
    if (!fileInfo.exists()) {
        QMessageBox::warning(this, "Error", "Image file not found: " + imagePath);
        return;
    }
    
    m_originalPixmap = QPixmap(imagePath);
    if (m_originalPixmap.isNull()) {
        QMessageBox::warning(this, "Error", "Could not load image: " + imagePath);
        return;
    }
    
    setWindowTitle("Image Viewer - " + fileInfo.fileName());
    onFitToWindow();
}

void ImageViewerDialog::onZoomIn()
{
    m_scaleFactor *= 1.25;
    updateImage();
}

void ImageViewerDialog::onZoomOut()
{
    m_scaleFactor /= 1.25;
    updateImage();
}

void ImageViewerDialog::onFitToWindow()
{
    if (m_originalPixmap.isNull() || !m_scrollArea) return;
    
    QSize scrollAreaSize = m_scrollArea->size();
    if (scrollAreaSize.width() <= 0 || scrollAreaSize.height() <= 0) {
        // If scroll area isn't sized yet, use a default scale
        m_scaleFactor = 1.0;
        updateImage();
        return;
    }
    
    QSize imageSize = m_originalPixmap.size();
    
    double scaleX = static_cast<double>(scrollAreaSize.width() - 20) / imageSize.width();
    double scaleY = static_cast<double>(scrollAreaSize.height() - 20) / imageSize.height();
    
    m_scaleFactor = qMin(scaleX, scaleY);
    updateImage();
}

void ImageViewerDialog::onClose()
{
    accept();
}

void ImageViewerDialog::updateImage()
{
    if (m_originalPixmap.isNull() || !m_imageLabel) return;
    
    QSize newSize = m_originalPixmap.size() * m_scaleFactor;
    QPixmap scaledPixmap = m_originalPixmap.scaled(newSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    
    m_imageLabel->setPixmap(scaledPixmap);
    m_imageLabel->adjustSize();
}
