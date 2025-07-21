use kairei::base_model::{
    BaseModel, BaseModelId, BaseModelMetadata, BaseModelRepository, PostgresBaseModelRepository,
};
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
    sqlx::query("DELETE FROM base_models WHERE name LIKE 'test_%'")
        .execute(&pool)
        .await
        .expect("Failed to clean test data");

    pool
}

fn create_test_model(name: &str) -> BaseModel {
    BaseModel {
        id: BaseModelId::new(),
        name: name.to_string(),
        description: Some("Test model description".to_string()),
        repo_id: Some("test/repo".to_string()),
        filename: Some("model.safetensors".to_string()),
        size_mb: Some(100),
        metadata: Some(BaseModelMetadata {
            repo_id: "test/repo".to_string(),
            name: name.to_string(),
            description: Some("Test metadata".to_string()),
            downloaded_at: Some(chrono::Utc::now().to_rfc3339()),
            parameters: Some("1B".to_string()),
            architecture: Some("transformer".to_string()),
            quantization: Some("fp16".to_string()),
        }),
    }
}

#[tokio::test]
async fn test_create_and_get() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool);

    // Create a model
    let model = create_test_model("test_create_get");
    let created = repo.create(model.clone()).await.unwrap();

    assert_eq!(created.name, model.name);
    assert_eq!(created.description, model.description);

    // Get the model by ID
    let retrieved = repo.get(&created.id).await.unwrap();
    assert_eq!(retrieved.name, created.name);
    assert_eq!(retrieved.repo_id, created.repo_id);
}

#[tokio::test]
async fn test_get_by_name() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool);

    // Create a model
    let model = create_test_model("test_get_by_name");
    let created = repo.create(model).await.unwrap();
    println!("Created model with name: {}", created.name);

    // Get by name
    match repo.get_by_name(&created.name).await {
        Ok(retrieved) => {
            assert_eq!(retrieved.id, created.id);
            assert_eq!(retrieved.name, created.name);
        }
        Err(e) => {
            panic!("Failed to get by name '{}': {:?}", created.name, e);
        }
    }
}

#[tokio::test]
async fn test_exists() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool);

    // Create a model
    let model = create_test_model("test_exists");
    let created = repo.create(model).await.unwrap();

    // Check exists
    assert!(repo.exists(&created.id).await.unwrap());

    // Check non-existent
    let fake_id = BaseModelId::new();
    assert!(!repo.exists(&fake_id).await.unwrap());
}

#[tokio::test]
async fn test_exists_by_name() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool);

    // Create a model
    let model = create_test_model("test_exists_by_name");
    repo.create(model).await.unwrap();

    // Check exists
    assert!(repo.exists_by_name("test_exists_by_name").await.unwrap());
    assert!(!repo.exists_by_name("non_existent").await.unwrap());
}

#[tokio::test]
async fn test_update() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool);

    // Create a model
    let model = create_test_model("test_update");
    let created = repo.create(model).await.unwrap();

    // Update the model
    let mut updated_model = created.clone();
    updated_model.description = Some("Updated description".to_string());
    updated_model.size_mb = Some(200);

    let updated = repo.update(updated_model).await.unwrap();
    assert_eq!(updated.description.as_deref(), Some("Updated description"));
    assert_eq!(updated.size_mb, Some(200));

    // Verify the update
    let retrieved = repo.get(&created.id).await.unwrap();
    assert_eq!(
        retrieved.description.as_deref(),
        Some("Updated description")
    );
}

#[tokio::test]
async fn test_delete() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool);

    // Create a model
    let model = create_test_model("test_delete");
    let created = repo.create(model).await.unwrap();

    // Delete the model
    repo.delete(&created.id).await.unwrap();

    // Verify deletion
    assert!(!repo.exists(&created.id).await.unwrap());
}

#[tokio::test]
async fn test_list() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool.clone());

    // Clean all test data first
    sqlx::query("DELETE FROM base_models WHERE name LIKE 'test_list_%'")
        .execute(&pool)
        .await
        .unwrap();

    // Create multiple models
    let models = vec![
        create_test_model("test_list_1"),
        create_test_model("test_list_2"),
        create_test_model("test_list_3"),
    ];

    for model in models {
        repo.create(model).await.unwrap();
    }

    // List all models
    let all_models = repo.list().await.unwrap();
    println!("Total models in DB: {}", all_models.len());

    let test_models: Vec<_> = all_models
        .into_iter()
        .filter(|m| {
            let matches = m.name.starts_with("test_list_");
            if matches {
                println!("Found test model: {}", m.name);
            }
            matches
        })
        .collect();

    println!("Test models found: {}", test_models.len());
    assert_eq!(test_models.len(), 3);
}

#[tokio::test]
async fn test_duplicate_name_error() {
    let pool = setup_test_db().await;
    let repo = PostgresBaseModelRepository::new(pool);

    // Create a model
    let model = create_test_model("test_duplicate");
    repo.create(model.clone()).await.unwrap();

    // Try to create another with same name
    let duplicate = create_test_model("test_duplicate");
    let result = repo.create(duplicate).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        kairei::base_model::BaseModelError::AlreadyExists(name) => {
            assert_eq!(name, "test_duplicate");
        }
        _ => panic!("Expected AlreadyExists error"),
    }
}
