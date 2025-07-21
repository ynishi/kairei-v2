use kairei::base_model::{
    BaseModel, BaseModelId, BaseModelRepository, PostgresBaseModelRepository,
};
use kairei::lora::{Lora, LoraMetadata, LoraRepository, LoraStatus, PostgresLoraRepository};
use sqlx::postgres::{PgPool, PgPoolOptions};

async fn setup_test_db() -> PgPool {
    dotenv::dotenv().ok();

    let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
        "postgresql://kairei_user:kairei_password@localhost:5432/kairei_dev".to_string()
    });

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("Failed to connect to database");

    // Run migrations
    sqlx::migrate!("../../migrations")
        .run(&pool)
        .await
        .expect("Failed to run migrations");

    // Clean up test data
    sqlx::query("DELETE FROM loras WHERE name LIKE 'test_%'")
        .execute(&pool)
        .await
        .expect("Failed to clean test loras");

    sqlx::query("DELETE FROM base_models WHERE name LIKE 'test_%'")
        .execute(&pool)
        .await
        .expect("Failed to clean test base models");

    pool
}

fn create_test_lora(name: &str) -> Lora {
    let unique_name = format!("{}_{}", name, uuid::Uuid::new_v4());
    let metadata = LoraMetadata {
        rank: Some(8),
        alpha: Some(16.0),
        training_info: None,
        parent_lora_id: None,
        version: Some("1.0.0".to_string()),
        training_framework: Some("transformers".to_string()),
    };

    Lora::new(
        unique_name,
        Some("Test LoRA description".to_string()),
        None, // Set to None or use a valid UUID
        metadata,
    )
}

async fn create_test_lora_with_base_model(pool: &PgPool, name: &str) -> Lora {
    // First create a base model
    let base_model_repo = PostgresBaseModelRepository::new(pool.clone());
    let unique_suffix = uuid::Uuid::new_v4();
    let base_model = BaseModel {
        id: BaseModelId::new(),
        name: format!("{}_base_{}", name, unique_suffix),
        description: Some("Test base model".to_string()),
        repo_id: Some("test/repo".to_string()),
        filename: Some("model.safetensors".to_string()),
        size_mb: Some(100),
        metadata: None,
    };
    let created_base = base_model_repo.create(base_model).await.unwrap();

    let unique_lora_name = format!("{}_{}", name, unique_suffix);
    let metadata = LoraMetadata {
        rank: Some(8),
        alpha: Some(16.0),
        training_info: None,
        parent_lora_id: None,
        version: Some("1.0.0".to_string()),
        training_framework: Some("transformers".to_string()),
    };

    Lora::new(
        unique_lora_name,
        Some("Test LoRA description".to_string()),
        Some(created_base.id),
        metadata,
    )
}

#[tokio::test]
async fn test_create_and_get() {
    let pool = setup_test_db().await;
    let repo = PostgresLoraRepository::new(pool.clone());

    // Create a LoRA
    let lora = create_test_lora("test_create_get");
    let created = repo.create(lora.clone()).await.unwrap();

    assert_eq!(created.name, lora.name);
    assert_eq!(created.description, lora.description);
    assert_eq!(created.status, LoraStatus::Available);
    assert!(!created.archived);

    // Get the LoRA by ID
    let retrieved = repo.get(&created.id).await.unwrap().unwrap();
    assert_eq!(retrieved.name, created.name);
    assert_eq!(retrieved.base_model_id, created.base_model_id);
    assert_eq!(retrieved.metadata.rank, Some(8));
    assert_eq!(retrieved.metadata.alpha, Some(16.0));

    // Test with base model
    let lora_with_base = create_test_lora_with_base_model(&pool, "test_create_get_with_base").await;
    let created_with_base = repo.create(lora_with_base.clone()).await.unwrap();

    assert!(created_with_base.base_model_id.is_some());
    assert_eq!(
        created_with_base.base_model_id,
        lora_with_base.base_model_id
    );

    // Get the LoRA with base model by ID
    let retrieved_with_base = repo.get(&created_with_base.id).await.unwrap().unwrap();
    assert_eq!(
        retrieved_with_base.base_model_id,
        created_with_base.base_model_id
    );
}

#[tokio::test]
async fn test_get_by_name() {
    let pool = setup_test_db().await;
    let repo = PostgresLoraRepository::new(pool.clone());

    // Create a LoRA
    let lora = create_test_lora("test_get_by_name");
    let created = repo.create(lora).await.unwrap();

    // Get by name
    let retrieved = repo.get_by_name(&created.name).await.unwrap().unwrap();
    assert_eq!(retrieved.id, created.id);
    assert_eq!(retrieved.name, created.name);
    assert_eq!(retrieved.status, LoraStatus::Available);

    // Test non-existent name
    let not_found = repo.get_by_name("non_existent").await.unwrap();
    assert!(not_found.is_none());
}

#[tokio::test]
async fn test_list() {
    let pool = setup_test_db().await;
    let repo = PostgresLoraRepository::new(pool.clone());

    // Clean all test data first
    sqlx::query("DELETE FROM loras WHERE name LIKE 'test_list_%'")
        .execute(&pool)
        .await
        .unwrap();

    // Create multiple LoRAs
    let loras = vec![
        create_test_lora("test_list_1"),
        create_test_lora("test_list_2"),
        create_test_lora("test_list_3"),
    ];

    for lora in loras {
        repo.create(lora).await.unwrap();
    }

    // Create an archived LoRA (should not appear in list)
    let mut archived_lora = create_test_lora("test_list_archived");
    archived_lora.archived = true;
    repo.create(archived_lora).await.unwrap();

    // List all non-archived LoRAs
    let all_loras = repo.list().await.unwrap();
    let test_loras: Vec<_> = all_loras
        .into_iter()
        .filter(|l| l.name.starts_with("test_list_"))
        .collect();

    assert_eq!(test_loras.len(), 3);
    assert!(test_loras.iter().all(|l| !l.archived));
    assert!(test_loras.iter().all(|l| l.status == LoraStatus::Available));
}

#[tokio::test]
async fn test_update() {
    let pool = setup_test_db().await;
    let repo = PostgresLoraRepository::new(pool.clone());

    // Create a LoRA
    let lora = create_test_lora("test_update");
    let created = repo.create(lora).await.unwrap();

    // Update the LoRA
    let mut updated_lora = created.clone();
    updated_lora.description = Some("Updated description".to_string());
    updated_lora.status = LoraStatus::Training;
    updated_lora.file_path = Some("/path/to/updated/adapter.safetensors".to_string());
    updated_lora.size_bytes = Some(1024 * 1024); // 1MB
    updated_lora.metadata.rank = Some(16);
    updated_lora.metadata.alpha = Some(32.0);
    updated_lora.updated_at = chrono::Utc::now().to_rfc3339();

    let updated = repo.update(updated_lora).await.unwrap();
    assert_eq!(updated.description.as_deref(), Some("Updated description"));
    assert_eq!(updated.status, LoraStatus::Training);
    assert_eq!(
        updated.file_path.as_deref(),
        Some("/path/to/updated/adapter.safetensors")
    );
    assert_eq!(updated.size_bytes, Some(1024 * 1024));

    // Verify the update by fetching again
    let retrieved = repo.get(&created.id).await.unwrap().unwrap();
    assert_eq!(
        retrieved.description.as_deref(),
        Some("Updated description")
    );
    assert_eq!(retrieved.status, LoraStatus::Training);
    assert_eq!(retrieved.metadata.rank, Some(16));
    assert_eq!(retrieved.metadata.alpha, Some(32.0));
}

#[tokio::test]
async fn test_delete() {
    let pool = setup_test_db().await;
    let repo = PostgresLoraRepository::new(pool.clone());

    // Create a LoRA
    let lora = create_test_lora("test_delete");
    let created = repo.create(lora).await.unwrap();

    // Delete the LoRA
    repo.delete(&created.id).await.unwrap();

    // Verify deletion
    let deleted = repo.get(&created.id).await.unwrap();
    assert!(deleted.is_none());

    // Try to delete non-existent LoRA
    let result = repo.delete(&created.id).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_duplicate_name_error() {
    let pool = setup_test_db().await;
    let repo = PostgresLoraRepository::new(pool.clone());

    // Create a LoRA with a fixed name (not unique)
    let fixed_name = format!("test_duplicate_{}", uuid::Uuid::new_v4());
    let metadata = LoraMetadata {
        rank: Some(8),
        alpha: Some(16.0),
        training_info: None,
        parent_lora_id: None,
        version: Some("1.0.0".to_string()),
        training_framework: Some("transformers".to_string()),
    };

    let lora1 = Lora::new(
        fixed_name.clone(),
        Some("Test LoRA description".to_string()),
        None,
        metadata.clone(),
    );
    repo.create(lora1).await.unwrap();

    // Try to create another with same name
    let lora2 = Lora::new(
        fixed_name.clone(),
        Some("Another LoRA description".to_string()),
        None,
        metadata,
    );
    let result = repo.create(lora2).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        kairei::lora::LoraError::AlreadyExists(name) => {
            assert_eq!(name, fixed_name);
        }
        _ => panic!("Expected AlreadyExists error"),
    }
}
