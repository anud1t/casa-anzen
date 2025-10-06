#pragma once

#include <QDialog>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QPixmap>

class ImageViewerDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ImageViewerDialog(const QString& imagePath, QWidget* parent = nullptr);
    ~ImageViewerDialog() = default;

private:
    void setupUI();
    void applyMilitaryTheme();
    void loadImage(const QString& imagePath);
    void updateImage();

    QVBoxLayout* m_mainLayout;
    QHBoxLayout* m_buttonLayout;
    QScrollArea* m_scrollArea;
    QLabel* m_imageLabel;
    QPushButton* m_closeButton;
    QPushButton* m_zoomInButton;
    QPushButton* m_zoomOutButton;
    QPushButton* m_fitButton;
    
    QPixmap m_originalPixmap;
    double m_scaleFactor;
    
private slots:
    void onZoomIn();
    void onZoomOut();
    void onFitToWindow();
    void onClose();
};
