
CREATE TABLE "embeddings" (
    "id" BIGINT GENERATED BY DEFAULT AS IDENTITY NOT NULL,
    "text" TEXT NOT NULL,
    "embedding_vector" VECTOR(1536) NOT NULL,
    "content_type" TEXT,
    "content_id" BIGINT,
    CONSTRAINT "embeddings_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "statistics" (
    "statistic_id" BIGINT NOT NULL,
    "page_title" TEXT NOT NULL,
    "catchline" TEXT NOT NULL,
    "graph_header" TEXT NOT NULL,
    "data" JSONB,
    CONSTRAINT "statistics_pkey" PRIMARY KEY ("statistic_id")
);

CREATE TABLE "statistics_embeddings" (
    "id" BIGINT GENERATED BY DEFAULT AS IDENTITY NOT NULL,
    "embedding_id" BIGINT NOT NULL,
    "statistic_id" BIGINT NOT NULL,
    CONSTRAINT "statistics_embeddings_pkey" PRIMARY KEY ("id")

);


ALTER TABLE "statistics_embeddings" ADD CONSTRAINT "statistics_embeddings_embedding_id_fkey" FOREIGN KEY ("embedding_id") REFERENCES "embeddings" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION;

ALTER TABLE "statistics_embeddings" ADD CONSTRAINT "statistics_embeddings_statistic_id_fkey" FOREIGN KEY ("statistic_id") REFERENCES "statistics" ("statistic_id") ON UPDATE NO ACTION ON DELETE NO ACTION;
