import Redis from "ioredis";
import pg from "pg";
import { S3Client } from "@aws-sdk/client-s3";

export function createRuntimeConnections({
  redisUrl,
  pgHost,
  pgPort,
  pgUser,
  pgPassword,
  pgDatabase,
  minioEndpoint,
  minioAccessKey,
  minioSecretKey,
}) {
  const redis = new Redis(redisUrl);
  const pool = new pg.Pool({
    host: pgHost,
    port: Number(pgPort || 5432),
    user: pgUser,
    password: pgPassword,
    database: pgDatabase,
  });
  const s3 = new S3Client({
    endpoint: minioEndpoint,
    credentials: {
      accessKeyId: minioAccessKey,
      secretAccessKey: minioSecretKey,
    },
    region: "us-east-1",
    forcePathStyle: true,
  });

  return { redis, pool, s3 };
}
