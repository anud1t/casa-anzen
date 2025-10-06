#pragma once

#include <QFrame>
#include <QLabel>
#include <QVBoxLayout>
#include <QPixmap>

class EventCard : public QFrame
{
    Q_OBJECT

public:
    explicit EventCard(QWidget* parent = nullptr);
    ~EventCard() = default;

    void setTitle(const QString& title);
    void setThumbnail(const QPixmap& thumbnail);
    void setCaption(const QString& caption);
    void setCaptionVisible(bool visible);
    void setSelected(bool selected);
    void setCaptionOverlayMode(bool overlay);
    void toggleCaptionOverlay();
    void setAICaption(const QString& caption);
    bool hasAICaption() const { return m_hasAICaption; }

    QString getTitle() const;
    QPixmap getThumbnail() const;  // Returns full resolution image
    QPixmap getDisplayThumbnail() const;  // Returns uniform-sized thumbnail
    QString getCaption() const;

signals:
    void clicked();
    void captionRequested();

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void setupUI();
    void applyMilitaryTheme();

    QVBoxLayout* m_layout;
    QLabel* m_titleBadge;
    QLabel* m_thumbnail;
    QLabel* m_captionLabel;
    
    QString m_title;
    QPixmap m_thumbnailPixmap;  // Full resolution image
    QPixmap m_displayThumbnail; // Uniform-sized thumbnail for display
    QString m_caption;
    QString m_baseStyleSheet;
    bool m_captionOverlayMode;  // Whether caption is overlaying the thumbnail
    bool m_hasAICaption;        // Whether this card has an AI-generated caption
    
    // Standard thumbnail size for uniform display
    static const QSize THUMBNAIL_SIZE;
};
