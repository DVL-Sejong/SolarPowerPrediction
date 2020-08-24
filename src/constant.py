import enum


class FeatureType(enum.Enum):
    TEMPERATURE = "기온(°C)"
    PRECIPITATION = "강수량(mm)"
    WIND_SPEED = "풍속(m/s)"
    WIND_DIRECTION = "풍향(16방위)"
    HUMIDITY = "습도(%)"
    DEW_POINT_TEMPERATURE = "이슬점온도(°C)"
    STEAM_PRESSURE = "현지기압(hPa)"
    SUNSHINE = "일조(hr)"
    VISIBILITY = "시정(10m)"
    GROUND_TEMPERATURE = "지면온도(°C)"
    ATMOSPHERIC_PRESSURE = "증기압(hPa)"


class FileType(enum.Enum):
    MODEL = 0
    RESULT = 1


class DataType(enum.Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 1
