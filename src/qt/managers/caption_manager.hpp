#pragma once

#include <QObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QBuffer>
#include <QImage>
#include <QListWidgetItem>

class CaptionManager : public QObject
{
    Q_OBJECT

public:
    explicit CaptionManager(QObject* parent = nullptr);
    ~CaptionManager() = default;

    void requestCaption(QListWidgetItem* item, const QString& imagePath);
    void setCaptionEndpoint(const QString& endpoint);

signals:
    void captionReady(QListWidgetItem* item, const QString& caption);
    void captionFailed(QListWidgetItem* item, const QString& error);

private slots:
    void onCaptionReplyFinished();

private:
    QString createDataUrl(const QString& imagePath);
    QString extractCaptionFromResponse(const QByteArray& response);
    void handleCaptionError(QListWidgetItem* item, const QString& error);

    QNetworkAccessManager* m_networkManager;
    QString m_captionEndpoint;
    QMap<QNetworkReply*, QListWidgetItem*> m_pendingRequests;
};
