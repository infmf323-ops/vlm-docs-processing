export function getStatusLabel(status: string): string {
  switch (status) {
    case "queued":
      return "В очереди";
    case "processing":
      return "Обработка";
    case "done":
      return "Готово";
    case "failed":
      return "Ошибка";
    default:
      return status;
  }
}

export function getFriendlyJobError(errorMessage?: string | null): string | null {
  if (!errorMessage) return null;

  const text = errorMessage.toLowerCase();

  if (text.includes("could not") && text.includes("extract")) {
    return "Модель не смогла извлечь данные из документа. Попробуйте другой файл или повторный запуск.";
  }

  if (text.includes("не смогла извлечь поля")) {
    return "Модель не смогла извлечь данные из документа. Попробуйте другой файл или повторный запуск.";
  }

  if (text.includes("зациклилась при генерации") || text.includes("loop") || text.includes("repetition")) {
    return "Модель зациклилась при генерации ответа. Попробуйте повторить запуск или выбрать другой документ.";
  }

  if (text.includes("supports only invoice extraction")) {
    return "Текущий движок Donut пока поддерживает только обработку счетов и invoice-документов.";
  }

  if (text.includes("not wired into the pipeline yet")) {
    return "Выбранный движок уже добавлен в архитектуру, но пока не подключен к полному pipeline.";
  }

  if (text.includes("schema_mapping_not_implemented")) {
    return "Текст документа извлечен, но структурный парсинг полей для этого типа документа пока не реализован.";
  }

  if (text.includes("identity-document parser found too few fields")) {
    return "Текст документа считан, но автоматический разбор полей сработал неуверенно.";
  }

  if (text.includes("unsupported operand type")) {
    return "Произошла внутренняя ошибка обработки. Попробуйте повторный запуск.";
  }

  if (text.includes("not found")) {
    return "Запрошенная job не найдена.";
  }

  return errorMessage;
}

export function getFieldLabel(key: string): string {
  switch (key) {
    case "invoice_no":
      return "Номер счета";
    case "invoice_date":
      return "Дата счета";
    case "due_date":
      return "Срок оплаты";
    case "currency":
      return "Валюта";
    case "seller":
      return "Продавец";
    case "buyer":
      return "Покупатель";
    case "seller_tax_id":
      return "ИНН продавца";
    case "client_tax_id":
      return "ИНН покупателя";
    case "iban":
      return "IBAN";
    case "line_items":
      return "Позиции";
    case "total_net_worth":
    case "total_net":
      return "Сумма без НДС";
    case "total_vat":
    case "total_tax":
      return "НДС";
    case "total_gross_worth":
    case "total_gross":
      return "Сумма с НДС";
    case "document_number":
      return "Номер документа";
    case "license_number":
      return "Номер удостоверения";
    case "surname":
      return "Фамилия";
    case "given_names":
      return "Имя";
    case "nationality":
      return "Гражданство";
    case "date_of_birth":
      return "Дата рождения";
    case "sex":
      return "Пол";
    case "place_of_birth":
      return "Место рождения";
    case "date_of_issue":
      return "Дата выдачи";
    case "date_of_expiry":
      return "Дата окончания";
    case "issuing_authority":
      return "Орган выдачи";
    case "mrz":
      return "MRZ";
    case "address":
      return "Адрес";
    case "personal_number":
      return "Персональный номер";
    case "categories":
      return "Категории";
    case "statement_period_start":
      return "Период с";
    case "statement_period_end":
      return "Период по";
    case "account_holder":
      return "Владелец счета";
    case "account_number":
      return "Номер счета";
    case "bank_name":
      return "Банк";
    case "opening_balance":
      return "Начальный баланс";
    case "closing_balance":
      return "Конечный баланс";
    case "extracted_pairs":
      return "Извлеченные пары";
    default:
      return key;
  }
}

export function getDocumentTypeLabel(value?: string | null): string {
  switch (value) {
    case "invoice":
      return "Счет / Invoice";
    case "passport":
      return "Паспорт";
    case "id_card":
      return "ID-карта";
    case "driver_license":
      return "Водительское удостоверение";
    case "financial_statement":
      return "Финансовая выписка";
    case "other":
      return "Другой документ";
    default:
      return value ?? "—";
  }
}

export function getEngineLabel(value?: string | null): string {
  switch (value) {
    case "donut":
      return "Donut";
    case "paddleocr_vl":
      return "PaddleOCR-VL";
    case "qwen2_5_vl":
      return "Qwen2.5-VL";
    case "custom":
      return "Custom";
    default:
      return value ?? "—";
  }
}
