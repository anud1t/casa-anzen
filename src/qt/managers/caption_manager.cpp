#include "caption_manager.hpp"
#include <QNetworkRequest>
#include <QFile>
#include <QDebug>

CaptionManager::CaptionManager(QObject* parent)
    : QObject(parent)
    , m_networkManager(new QNetworkAccessManager(this))
    , m_captionEndpoint("http://localhost:2020/v1/caption")
{
    connect(m_networkManager, &QNetworkAccessManager::finished,
            this, &CaptionManager::onCaptionReplyFinished);
}

void CaptionManager::requestCaption(QListWidgetItem* item, const QString& imagePath)
{
    if (!item || imagePath.isEmpty()) {
        emit captionFailed(item, "Invalid item or image path");
        return;
    }

    QString dataUrl = createDataUrl(imagePath);
    if (dataUrl.isEmpty()) {
        emit captionFailed(item, "Failed to create data URL");
        return;
    }

    QJsonObject payload;
    payload.insert("image_url", dataUrl);
    payload.insert("length", "short");
    
    QJsonDocument doc(payload);
    QByteArray body = doc.toJson(QJsonDocument::Compact);

    QNetworkRequest request{QUrl(m_captionEndpoint)};
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply* reply = m_networkManager->post(request, body);
    m_pendingRequests[reply] = item;
}

void CaptionManager::setCaptionEndpoint(const QString& endpoint)
{
    m_captionEndpoint = endpoint;
}

QString CaptionManager::createDataUrl(const QString& imagePath)
{
    QImage image(imagePath);
    if (image.isNull()) {
        return QString();
    }

    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    
    if (!image.save(&buffer, "JPEG")) {
        return QString();
    }

    QByteArray base64 = bytes.toBase64();
    return QStringLiteral("data:image/jpeg;base64,") + QString::fromLatin1(base64);
}

QString CaptionManager::extractCaptionFromResponse(const QByteArray& response)
{
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(response, &error);
    
    if (error.error != QJsonParseError::NoError || !doc.isObject()) {
        return QString();
    }

    QJsonObject obj = doc.object();
    QString caption;
    
    if (obj.contains("caption")) {
        caption = obj.value("caption").toString();
    } else if (obj.contains("data") && obj.value("data").isObject()) {
        caption = obj.value("data").toObject().value("caption").toString();
    }

    return caption.isEmpty() ? "(no caption)" : caption;
}

void CaptionManager::handleCaptionError(QListWidgetItem* item, const QString& error)
{
    qDebug() << "Caption error for item:" << error;
    emit captionFailed(item, error);
}

void CaptionManager::onCaptionReplyFinished()
{
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(sender());
    if (!reply) return;

    QListWidgetItem* item = m_pendingRequests.take(reply);
    if (!item) {
        reply->deleteLater();
        return;
    }

    if (reply->error() != QNetworkReply::NoError) {
        handleCaptionError(item, "Network error: " + reply->errorString());
        reply->deleteLater();
        return;
    }

    QByteArray response = reply->readAll();
    QString caption = extractCaptionFromResponse(response);
    
    if (caption.isEmpty()) {
        emit captionFailed(item, "Failed to parse caption response");
    } else {
        emit captionReady(item, caption);
    }

    reply->deleteLater();
}
