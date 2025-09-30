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

    QString getTitle() const;
    QPixmap getThumbnail() const;
    QString getCaption() const;

signals:
    void clicked();
    void captionRequested();

protected:
    void mousePressEvent(QMouseEvent* event) override;

private:
    void setupUI();
    void applyMilitaryTheme();
    void updateCaptionSizing();

    QVBoxLayout* m_layout;
    QLabel* m_titleBadge;
    QLabel* m_thumbnail;
    QLabel* m_captionLabel;
    
    QString m_title;
    QPixmap m_thumbnailPixmap;
    QString m_caption;
};
